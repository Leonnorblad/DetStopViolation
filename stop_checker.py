from config import *

def get_first_frame(video_name):
    video_path = f"videos/{video_name}.mp4"
    frames_generator = sv.get_video_frames_generator(source_path=video_path)
    iterator = iter(frames_generator)
    first_frame = next(iterator)
    os.makedirs(f"first_frames/", exist_ok=True)
    cv2.imwrite(f"first_frames/{video_name}.png", first_frame)

class StopVideo:
    def __init__(self, video_name):
        self.video_name = video_name
        self.output_path = f"output/{video_name}"
        os.makedirs(self.output_path, exist_ok=True)

        # Load video data
        self.video_info, self.frames_generator, stopzone, outzone, self.point, road = self.load_video(self.video_name)
        # Load detection models
        self.model_wheels = get_model("wheels-detection-fgbtv/1",
                                      api_key=os.environ["ROBOFLOW_API_KEY"])
        self.model_yolo = get_model("yolov8s-640")
    
        # Initialize trackers
        self.tracker_yolo = sv.ByteTrack(frame_rate=self.video_info.fps)
        self.tracker_wheel = sv.ByteTrack(frame_rate=self.video_info.fps)

        # Box annotators
        self.box_annotator_green = sv.BoxAnnotator(color=sv.Color.GREEN)
        self.box_annotator_gray = sv.BoxAnnotator(color=sv.Color(r=176, g=178, b=181))
        self.box_annotator_red = sv.BoxAnnotator(color=sv.Color.RED)
        self.box_annotator_white = sv.BoxAnnotator(color=sv.Color.WHITE)

        # Label annotators
        scale = 1.05
        self.label_annotator_green = sv.LabelAnnotator(color=sv.Color.GREEN, text_scale=scale)
        self.label_annotator_gray = sv.LabelAnnotator(color=sv.Color(r=176, g=178, b=181), text_scale=scale)
        self.label_annotator_red = sv.LabelAnnotator(color=sv.Color.RED, text_scale=scale)

        # Zones
        self.stopzone, self.stopzone_annotator = self.add_zone(sv.Color.GREEN, stopzone)
        self.outzone, self.outzone_annotator = self.add_zone(sv.Color.RED, outzone)
        self.road, self.road_annotator = self.add_zone(sv.Color.BLUE, ROAD[self.video_name])
        
    def inference(self):
        """
        Performs inference to get car and wheel detections.
        """
        # Bbox detections of car locations. Key is frame number, value is the detections
        self.car_detections = {}
        # Bbox detections of wheel locations. Key is frame number, value is the detections
        self.wheel_detections = {}
        # Report for each frame. Key is frame number, value is the report (created in in construct_report())
        self.report = {}
        # Center position of the car. Key is frame number, value is the center position. Used to calculate speed.
        self.car_center_history = {}
        self.first_last_frames = {}

        self.video_info, self.frames_generator, stopzone, outzone, self.point, road = self.load_video(self.video_name)
        
        for frame_no, frame in enumerate(tqdm(self.frames_generator, total=self.video_info.total_frames), start=1):
            #### 1. For veichle detection ####
            # Perform inference on the frame - This detects all objects defined by the model
            results_yolo = self.model_yolo.infer(frame)[0]
            detections_yolo_all = sv.Detections.from_inference(results_yolo)
            # Filter to only include vehicles
            detections_yolo_vehicle = detections_yolo_all[self.is_vehicle(detections_yolo_all)]

            # Update the tracker with the detections - to get ID of the vehicle
            detections_yolo_vehicle = self.tracker_yolo.update_with_detections(detections_yolo_vehicle)

            car_ids = self.get_ids(detections_yolo_vehicle)
            for id in car_ids:
                if id not in self.first_last_frames:
                    self.first_last_frames[id] = {"first": frame_no, "last": frame_no}
                else:
                    self.first_last_frames[id]["last"] = frame_no
            
            # Filter to only include vehicles on the road of the stop sign
            on_road_mask = self.road.trigger(detections_yolo_vehicle)
            detections_yolo_vehicle = detections_yolo_vehicle[on_road_mask]

            # Filter to only include the vehicle closest to the point
            detections_yolo_vehicle = self.return_target(detections_yolo_vehicle)
            
            
            # Extract the vehicle from the frame
            vehicle, vehicle_coords = self.extract_vehicle(frame, detections_yolo_vehicle)
            
            # Save car detections - if we have any
            if len(detections_yolo_vehicle) > 0:
                self.car_detections[frame_no] = detections_yolo_vehicle

            #### 2. For wheel detection ####
            # If there is a vehicle in the frame, we perform wheel detection
            if vehicle is not None:
                # Perform inference on the vehicle
                results_wheel = self.model_wheels.infer(np.array(vehicle))[0]
                detections_wheel = sv.Detections.from_inference(results_wheel)

                # If we found a wheel in the sliced image, we need to update the coordinates to the full image
                if len(detections_wheel) > 0:
                    detections_wheel = self.update_wheel_coords(detections_wheel, vehicle_coords)
                detections_wheel_id = self.tracker_wheel.update_with_detections(detections_wheel)
                # Save wheel detection if we have any
                if len(detections_wheel_id) > 0:
                    self.wheel_detections[frame_no] = detections_wheel_id

            # Construct frame report
            self.report[frame_no] = self.construct_report(frame_no)
                    
    def render_video_simple(self):
        self.video_info, self.frames_generator, _, _, _, _ = self.load_video(self.video_name)

        with sv.VideoSink(target_path=f"{self.output_path}/inference_simple.mp4", video_info=self.video_info) as sink:
            for frame_no, frame in enumerate(tqdm(self.frames_generator, total=self.video_info.total_frames), start=1):
                annotated_frame = frame.copy()

                if frame_no in self.car_detections:
                    car_detections = self.car_detections[frame_no]
                    
                    car_id = self.get_ids(car_detections)[0]
                    # Avoid to show cars that does not have a status (stopped or failed to stop)
                    # reported_wheels is a dict with car_id as key and the wheel_id that has been in the stopzone + outzone
                    if car_id not in self.reported_wheels:
                        sink.write_frame(annotated_frame)
                        continue
                    
                    label = self.add_label(frame_no)
                    color = self.get_color(label)
                    if color=="green":
                        annotated_frame = self.label_annotator_green.annotate(annotated_frame, car_detections, labels=label)
                        annotated_frame = self.box_annotator_green.annotate(annotated_frame, car_detections)
                    elif color=="red":
                        annotated_frame = self.label_annotator_red.annotate(annotated_frame, car_detections, labels=label)
                        annotated_frame = self.box_annotator_red.annotate(annotated_frame, car_detections)
                    else:
                        annotated_frame = self.label_annotator_gray.annotate(annotated_frame, car_detections, labels=label)
                        annotated_frame = self.box_annotator_gray.annotate(annotated_frame, car_detections)

                sink.write_frame(annotated_frame)

    def render_video_advanced(self):
        self.video_info, self.frames_generator, stopzone, outzone, self.point, self.road = self.load_video(self.video_name)
        with sv.VideoSink(target_path=f"{self.output_path}/inference_advanced.mp4", video_info=self.video_info) as sink:
            for frame_no, frame in enumerate(tqdm(self.frames_generator, total=self.video_info.total_frames), start=1):
                annotated_frame = frame.copy()
                annotated_frame = self.stopzone_annotator.annotate(scene=annotated_frame)
                annotated_frame = self.outzone_annotator.annotate(scene=annotated_frame)
                annotated_frame = self.road_annotator.annotate(scene=annotated_frame)

                # Plot the point
                point_x, point_y = self.point
                cv2.circle(annotated_frame, (int(point_x), int(point_y)), 10, (0, 255, 0), -1)

                if frame_no in self.car_detections:
                    car_detections = self.car_detections[frame_no]

                    # Extra check to only show the cars that has a status (stopped or failed to stop)
                    # car detections will always have max 1 detection. We do not want to display the car if it is not in reported_wheels

                    car_id = self.get_ids(car_detections)[0]
                    if car_id not in self.reported_wheels:
                        sink.write_frame(annotated_frame)
                        continue
                    
                    # Get the center coordinates of the car
                    car_center = self.get_car_center(frame_no)
                    cv2.circle(annotated_frame, (int(car_center[0]), int(car_center[1])), 10, (0, 255, 0), -1)

                    # Plot line from car center to point
                    line_color = (0, 255, 0)
                    line_thickness = 2
                    line_type = cv2.LINE_AA
                    cv2.line(annotated_frame, (int(point_x), int(point_y)), (int(car_center[0]), int(car_center[1])), line_color, line_thickness, line_type)
                    
                    # Get the label (on car)
                    label = self.add_label(frame_no)
                    color = self.get_color(label)
                    if color=="green":
                        annotated_frame = self.label_annotator_green.annotate(annotated_frame, car_detections, labels=label)
                        annotated_frame = self.box_annotator_green.annotate(annotated_frame, car_detections)
                    elif color=="red":
                        annotated_frame = self.label_annotator_red.annotate(annotated_frame, car_detections, labels=label)
                        annotated_frame = self.box_annotator_red.annotate(annotated_frame, car_detections)
                    else:
                        annotated_frame = self.label_annotator_gray.annotate(annotated_frame, car_detections, labels=label)
                        annotated_frame = self.box_annotator_gray.annotate(annotated_frame, car_detections)

                # Show wheels on car
                if frame_no in self.wheel_detections:
                    wheel_detections = self.wheel_detections[frame_no]
                    # Extra check to only show the wheels that has a status (stopped or failed to stop)
                    wheel_detections = self.filter_detections(wheel_detections, self.reported_wheels[car_id])

                    annotated_frame = self.box_annotator_white.annotate(annotated_frame, wheel_detections)
                
                
                sink.write_frame(annotated_frame)

    def analyze(self):
        wheels_in_stopzone = {}
        wheels_in_outzone = {}
        self.car_left_zone = {}
        self.car_speeds = {}
        stopped = {}
        car_id_history = []
        
        self.reported_wheels = {}
        self.analysis = {}
        
        for frame_id, frame_dict in self.report.items():
            for car_id, vehicle_dict in frame_dict.items():

                if car_id not in car_id_history:
                    car_id_history.append(car_id)

                if len(vehicle_dict)==0:
                    continue
                
                # Add the wheels in the stopzone
                if car_id not in wheels_in_stopzone:
                    wheels_in_stopzone[car_id] = []
                else:
                    for wheel_dict in vehicle_dict["wheel"]:
                        for wheel_id, wheel_info in wheel_dict.items():
                            if wheel_info["stopzone"] and wheel_id not in wheels_in_stopzone[car_id]:
                                wheels_in_stopzone[car_id].append(wheel_id)

                # Add the wheels in the outzone
                if car_id not in wheels_in_outzone:
                    wheels_in_outzone[car_id] = []
                else:
                    for wheel_dict in vehicle_dict["wheel"]:
                        for wheel_id, wheel_info in wheel_dict.items():
                            if wheel_info["outzone"] and wheel_id not in wheels_in_outzone[car_id]:
                                wheels_in_outzone[car_id].append(wheel_id)
                
                # Save speed if the car is in the stopzone 
                if len(wheels_in_stopzone[car_id])>0:
                    if car_id not in self.car_speeds:
                        self.car_speeds[car_id] = []
                    self.car_speeds[car_id].append({"frame_no":frame_id, "speed":vehicle_dict['speed']})
                
                # Detect if the car has a wheel that has been in the stopzone that is now in the outzone
                for wheel in wheels_in_stopzone[car_id]:
                    if wheel in wheels_in_outzone[car_id]:
                        if car_id not in self.reported_wheels:
                            self.reported_wheels[car_id] = wheel
                        if car_id not in stopped:
                            filtered_speeds = [x for x in self.car_speeds[car_id] if x["speed"] is not None]
                            min_speed_in_stopzone = min(filtered_speeds, key=lambda x: x['speed'])
                            if min_speed_in_stopzone["speed"] < 15: # TODO: add threshold to config
                                stopped[car_id] = True
                                self.car_left_zone[car_id] = min_speed_in_stopzone["frame_no"]
                            else:
                                stopped[car_id] = False
                                self.car_left_zone[car_id] = frame_id
        
        # Construct the analysis
        for car_id in car_id_history:
            status = "Could not detect" if car_id not in stopped else "Stopped" if stopped[car_id] else "Failed to stop"
            # Only add the car to the analysis if it has been in the stopzone and outzone
            if status != "Could not detect":
                self.analysis[car_id] = {"First Entrance": self.first_last_frames[car_id]["first"],
                                        "Last Exit": self.first_last_frames[car_id]["last"],
                                        "Status": status}
          
    def construct_report(self, frame_no):
        """
        Construct a summary report for a given frame.

        Args:
            frame_no (int): The frame number to construct the report for.

        Returns:
            dict: A nested dictionary with the following structure:
                {
                    car_id: {
                        "wheel": [
                            {
                                wheel_id: {
                                    "stopzone": bool,
                                    "outzone": bool
                                }
                            },
                            ...
                        ],
                        "speed": float,
                        "car_center": tuple
                    }
                }
                If no car is detected in the frame, returns an empty dictionary.
                If a car is detected but no wheels are detected, returns a dictionary with car_id and an empty dictionary for wheels.
        """
        # 1. If we dont have a car in frame
        if frame_no not in self.car_detections:
            return {}
        
        car_id = self.get_ids(self.car_detections[frame_no])[0]

        # 2. If we dont have a wheel in frame
        if frame_no not in self.wheel_detections:
            return {car_id:{}}

        # 3. If we have a car and a wheel in frame
        wheel_ids = self.get_ids(self.wheel_detections[frame_no])

        in_stopping_zone = self.stopzone.trigger(self.wheel_detections[frame_no])
        in_outzone = self.outzone.trigger(self.wheel_detections[frame_no])
        wheel_lst = []
        for wheel, stop, out in zip(wheel_ids, in_stopping_zone, in_outzone):
            wheel_dict = {wheel:{"stopzone":stop, "outzone": out}}
            wheel_lst.append(wheel_dict)

        ### Speed
        car_center = self.get_car_center(frame_no)
        if car_id not in self.car_center_history:
            self.car_center_history[car_id] = [None, car_center]
        else:
            self.car_center_history[car_id].append(car_center)
        
        if self.car_center_history[car_id][-2] is not None:
            distance_moved = self.get_distance(self.car_center_history[car_id][-2], car_center)
            current_speed = distance_moved * self.video_info.fps
        else:
            current_speed = None
        
        # Return the dict for the frame
        return {car_id:{"wheel":wheel_lst, "speed":current_speed, "car_center":car_center}}
    
    def filter_detections(self, detections, id):
        """
        Filters detections to only include the detections with the given id.
        Used to only show the front wheel.
        """
        boo = [True if detection[4] == id else False for detection in detections]
        return detections[boo]

    def get_color(self, label):
        """ Returns the color of the label """
        if "Stopped" in label[0]:
            return "green"
        elif "Failed" in label[0]:
            return "red"
        else:
            return "gray"

    def add_label(self, frame_no):
        """ Construct a label for a given frame."""
        # If we have a car in the frame
        if frame_no in self.car_detections:
            car_id = self.get_ids(self.car_detections[frame_no])[0]
            status = self.analysis[car_id]['Status'] # This is the label from the analysis (Stopped, Failed to stop, Could not detect)

            if car_id in self.car_left_zone and frame_no>=self.car_left_zone[car_id]:
                return [f"{status} (id:{car_id})"]
            else:
                return [f"Detected (id:{car_id})"]
        # If we dont have a car in the frame, we do not return any labels
    
    def plot_car_speed(self, ylim, type="all", save=False):
        # Preprocess before plotting
        car_speeds = {}
        car_frames = {}
        for car_id, speeds in self.car_speeds.items():
            for speed in speeds:
                if car_id not in car_speeds:
                    car_speeds[car_id] = []
                    car_frames[car_id] = []
                car_speeds[car_id].append(speed["speed"])
                car_frames[car_id].append(speed["frame_no"])
        # Plot all cars
        if type=="all":
            for track_id, _ in self.analysis.items():
                frames = car_frames[track_id]
                speeds = car_speeds[track_id]
                plt.plot(frames, speeds, label=f"Vehicle {track_id} - {self.analysis[track_id]['Status']}")
                plt.ylim(ylim)
            plt.xlabel("Frame")
            plt.ylabel("Speed (pixels/sec)")
            plt.legend()
            if save:
                plt.savefig(f"{self.output_path}/speed_plot.png")
            else:
                plt.show()
            plt.close()
            
        if type=="seperate":
            for track_id, _ in self.analysis.items():
                frames = car_frames[track_id]
                speeds = car_speeds[track_id]
                plt.plot(frames, speeds, label=f"Vehicle {track_id} - {self.analysis[track_id]['Status']}")
                plt.ylim(ylim)
                plt.xlabel("Frame")
                plt.ylabel("Speed (pixels/sec)")
                plt.legend()
                if save:
                    plt.savefig(f"{self.output_path}/speed_plot_{track_id}.png")
                else:
                    plt.show()
                plt.close()
            
    def get_distance(self, prev_center, current_center):
        """
        Returns the distance between two points.
        Used to calculate speed.
        """
        return ((prev_center[0]-current_center[0])**2 + (prev_center[1]-current_center[1])**2)**0.5

    def get_car_center(self, frame_no):
        """ Returns the center of the car in the frame """
        x1, y1, x2, y2 = self.car_detections[frame_no].xyxy[0]
        center = (x1+x2)/2, (y1+y2)/2
        return center
        
    def get_ids(self, detections):
        """ Returns the ids of the detections (car or wheel ids) """
        return [detection[4] for detection in detections]

    def add_zone(self, color, zone_coords):
        """Adds a zone to the video, used to detect if a vehicle is in the stopzone, outzone or road"""
        zone = sv.PolygonZone(zone_coords)
        zone_annotator = sv.PolygonZoneAnnotator(
            display_in_zone_count=False,
            zone=zone,
            color=color)
        return zone, zone_annotator
                
    def load_video(self, video_name):
        video_path = f"videos/{video_name}.mp4"
        video_info = sv.VideoInfo.from_video_path(video_path=video_path)
        frames_generator = sv.get_video_frames_generator(source_path=video_path)
        road = ROAD[video_name]
        stopzone = STOP_ZONE[video_name]
        point = POINT[video_name]
        outzone = OUT_ZONE[video_name]
        return video_info, frames_generator, stopzone, outzone, point, road
        
    def extract_vehicle(self, frame, detections):
        """
        Extracts the vehicle from the frame

        Args:
            frame: The frame to extract the vehicle from
            detections: The detections of the vehicle

        Returns:
            vehicle: The vehicle extracted from the frame
            vehicle_coords: The coordinates of the vehicle in the frame
        """
        if len(detections) == 0:
            return None, None
        # Cut the frame to the bounding box of the vehicle and return it. We will alwas one have one vehicle in the frame
        for detection in detections:
            # define the bounding box
            x1, y1, x2, y2 = detection[0]
            vehicle = frame[int(y1):int(y2), int(x1):int(x2)]
            vehicle_coords = (int(x1), int(y1), int(x2), int(y2))
            return vehicle, vehicle_coords

    def update_wheel_coords(self, wheel_detections, vehicle_coords):
        """ Adjusts the wheel coordinates to the full frame"""
        x1, y1, x2, y2 = vehicle_coords
        for i, xyxy in enumerate(wheel_detections.xyxy):
            w_x1, w_y1, w_x2, w_y2 = xyxy
            nw_x1 = w_x1 + x1
            nw_y1 = w_y1 + y1
            nw_x2 = w_x2 + x1
            nw_y2 = w_y2 + y1
            wheel_detections.xyxy[i] = [nw_x1, nw_y1, nw_x2, nw_y2]
        return wheel_detections
    
    def is_vehicle(self, detections):
        """ Filters detections to only include vehicles"""
        vehicle_class_ids = [2, 3, 5, 7] # ["car", "motorcycle", "bus", "truck"]
        return [True if x in vehicle_class_ids else False for x in detections.class_id]
        
    def return_target(self, detections):
        """
        Filters detections to only include the vehicle closest to the point.
        This is helpful if there is a queue of cars and we only want to track the car closest to the stopline.
        """
        if len(detections) == 0:
            return detections
        else:
            distances = []
            for detection in detections:
                distance_to_point = self.get_distance_to_point(detection)
                distances.append(distance_to_point)
            
            # Find the index of the detection with the minimum distance
            min_dist_index = np.argmin(distances)
            
            # Return the detection with the minimum distance
            closest_detection = detections[min_dist_index:min_dist_index+1]
            return closest_detection
    
    def get_distance_to_point(self, detection):
        """ Returns the distance from the center of the detection to the point """
        x1, y1, x2, y2 = detection[0]
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        return ((center[0] - self.point[0]) ** 2 + (center[1] - self.point[1]) ** 2) ** 0.5
    
    def save_not_stopping(self, num_xra_sec=1):
        """
        Saves one video snippet of each car that failed to stop.
        Uses self.analysis to find the car_id and the frame_no where the car entered and left the frame.
        The videos are saved under subfolders: /not_stopping/video_name/car_id.mp4
        """
        # Number of seconds to add
        num_xra_frames = self.video_info.fps * num_xra_sec

        for car_id, values in self.analysis.items():
            if values["Status"] == "Failed to stop":
                video_path = f"videos/{self.video_name}.mp4"
                video_info = sv.VideoInfo.from_video_path(video_path=video_path)
                frames_generator = sv.get_video_frames_generator(source_path=video_path)

                os.makedirs(f"{self.output_path}/not_stopping/", exist_ok=True)

                with sv.VideoSink(target_path=f"{self.output_path}/not_stopping/{car_id}.mp4", video_info=video_info) as sink:
                    for frame_no, frame in enumerate(frames_generator, start=1):
                        start_frame = values["First Entrance"]-num_xra_frames
                        end_frame = values["Last Exit"]+num_xra_frames
                        if frame_no >= start_frame and frame_no <= end_frame:
                            sink.write_frame(frame)
                        if frame_no > end_frame:
                            break