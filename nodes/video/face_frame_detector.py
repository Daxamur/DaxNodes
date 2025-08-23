"""
Face Frame Detection Node for ComfyUI

Advanced face detection and eye state analysis for optimal video frame selection.
Uses MediaPipe for robust face detection and custom EAR calculation for eye analysis.

Credits:
- MediaPipe face detection by Google (Apache License 2.0)
- Eye Aspect Ratio (EAR) algorithm for blink detection

Copyright (c) 2025 DaxNodes
Licensed under MIT License
"""

import torch
import numpy as np
import json
from ...utils.debug_utils import debug_print
from typing import Optional, Tuple, List, Dict
from pathlib import Path

class FaceFrameDetector:
    """Find optimal reference frame with face and open eyes for video continuation"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "detection_method": ([
                    "mediapipe",
                    "mediapipe_detailed", 
                    "bbox_fallback"
                ], {"default": "mediapipe"}),
                "min_face_size": ("INT", {
                    "default": 50,
                    "min": 20,
                    "max": 200,
                    "step": 10
                }),
                "eye_open_threshold": ("FLOAT", {
                    "default": 0.24,
                    "min": 0.15,
                    "max": 0.4,
                    "step": 0.01
                }),
                "face_confidence": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.3,
                    "max": 1.0,
                    "step": 0.05
                }),
            },
            "optional": {
                "scan_limit": ("INT", {
                    "default": 20,
                    "min": 5,
                    "max": 50,
                    "step": 1
                }),
                "require_frontal": ("BOOLEAN", {"default": True}),
                "prefer_center": ("BOOLEAN", {"default": True}),
                "debug_output": ("BOOLEAN", {"default": True, "tooltip": "Show debug bounding boxes and detection info"}),
                "enable_face_detection": ("BOOLEAN", {"default": True, "tooltip": "When false, skips detection and outputs last frame"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "STRING", "IMAGE")
    RETURN_NAMES = ("best_face_frame", "frames_from_end", "detection_metadata", "debug_overlay")
    FUNCTION = "detect_best_face_frame"
    CATEGORY = "Video"
    
    def __init__(self):
        try:
            import mediapipe as mp
            self.mp = mp
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.has_mediapipe = True
        except ImportError:
            self.has_mediapipe = False
            print("[FaceFrameDetector] MediaPipe not found. Install with: pip install mediapipe")
        
        try:
            import cv2
            self.cv2 = cv2
            self.has_cv2 = True
        except ImportError:
            self.has_cv2 = False
            print("[FaceFrameDetector] OpenCV not found. Some features limited.")
        
    def detect_best_face_frame(self, images, detection_method="mediapipe", 
                              min_face_size=50, eye_open_threshold=0.22, 
                              face_confidence=0.7, scan_limit=20, 
                              require_frontal=True, prefer_center=True,
                              debug_output=False, enable_face_detection=True):
        
        # If enable_face_detection is False, skip all detection and return last frame
        if not enable_face_detection:
            print("Face detection disabled, returning last frame")
            last_frame = images[-1]
            metadata = {
                "total_frames_scanned": 0,
                "best_frame_score": 1.0,
                "frames_from_end": 0,
                "detection_method": "skipped",
                "all_detections": [],
                "detection_settings": {
                    "forced": True
                }
            }
            debug_overlay = last_frame.unsqueeze(0) if last_frame.dim() == 3 else last_frame
            return (
                last_frame.unsqueeze(0) if last_frame.dim() == 3 else last_frame,
                0,  # Always output 0 for last frame
                json.dumps(metadata),
                debug_overlay
            )
        
        # Handle empty or invalid input
        if images is None or images.shape[0] == 0:
            print("ERROR: Empty or invalid image batch received")
            # Return safe defaults
            empty_frame = torch.zeros(1, 64, 64, 3) if images is None else images[:1]
            return (empty_frame, 0, json.dumps({"error": "empty_input"}), empty_frame)
        
        total_frames = images.shape[0]
        scan_frames = min(scan_limit, total_frames)
        
        best_frame = None
        best_frame_idx = -1
        best_score = 0.0
        detection_results = []
        debug_frames = []
        
        # Check last frame first (prioritize recency)
        last_frame = images[-1]
        last_frame_result = self._detect_face(last_frame, detection_method, min_face_size, 
                                             eye_open_threshold, face_confidence)
        
        if require_frontal and not last_frame_result["is_frontal"]:
            last_frame_result["score"] *= 0.5
        
        if prefer_center:
            center_bonus = self._calculate_center_bonus(last_frame_result, last_frame.shape)
            last_frame_result["score"] *= center_bonus
        
        # Scan backwards to find first frame where all faces have open eyes
        frames_scanned = 0
        fallback_frame = None
        fallback_idx = 0
        fallback_score = 0
        max_open_eyes = 0
        
        for i in range(scan_frames):
            if i >= total_frames:
                break
            
            frame_idx = total_frames - 1 - i
            
            # Skip first 5 frames to prevent loop potential
            if frame_idx < 5:
                debug_print(f"Skipping frame {frame_idx} (too close to beginning)")
                continue
                
            frames_scanned = i
            frame = images[frame_idx]
            
            if i == 0:
                # Use pre-calculated last frame result
                result = last_frame_result.copy()
            else:
                result = self._detect_face(frame, detection_method, min_face_size, 
                                          eye_open_threshold, face_confidence)
                
                if require_frontal and not result["is_frontal"]:
                    result["score"] *= 0.5
                
                if prefer_center:
                    center_bonus = self._calculate_center_bonus(result, frame.shape)
                    result["score"] *= center_bonus
            
            detection_results.append({
                "frame_idx": frame_idx,
                "frames_from_end": i,
                **result
            })
            
            if debug_output:
                debug_frame = self._create_debug_overlay(frame, result)
                debug_frames.append(debug_frame)
            
            # Check if ALL faces have open eyes
            if result["has_face"] and result["eyes_open"] and result["score"] > 0.3:
                # Perfect frame - all detected faces have open eyes
                best_frame = frame
                best_frame_idx = i  # Use loop index directly (0 = last frame)
                best_score = result["score"]
                print(f"Found perfect frame: {result.get('num_faces', 1)} faces, all eyes open")
                debug_print(f"Selected frame_idx={frame_idx} (i={i}), is_last_frame={frame_idx == total_frames - 1}")
                break  # Stop immediately when perfect frame is found
            
            # Track best fallback option (highest EAR for faces with closed eyes)
            elif result["has_face"] and result["score"] > 0.3:
                # Use frame with most open eyes (highest EAR) even if not fully open
                current_ear = result.get("eye_aspect_ratio", 0.0)
                if current_ear > fallback_score:  # fallback_score now tracks best EAR
                    fallback_frame = frame
                    fallback_idx = i
                    fallback_score = current_ear
                    debug_print(f"Potential fallback at frames_from_end={i}: EAR={current_ear:.3f} (most open eyes)")
        
        # Use fallback if no perfect frame found (frame with most open eyes)
        if best_frame is None and fallback_frame is not None:
            best_frame = fallback_frame
            best_frame_idx = fallback_idx
            best_score = fallback_score  # This is now the EAR value
            debug_print(f"Using fallback frame at frames_from_end={best_frame_idx} with highest EAR={best_score:.3f}")
        
        # Final fallback: when no faces found at all, use last frame
        if best_frame is None:
            print("No faces detected, using last frame")
            best_frame = images[-1]
            best_frame_idx = 0  # Last frame outputs 0
            best_score = last_frame_result.get("score", 0.0) if last_frame_result else 0.0
            debug_print(f"Final fallback: using images[-1] (last frame), frames_from_end=0")
        
        metadata = {
            "total_frames_scanned": scan_frames,
            "best_frame_score": best_score,
            "frames_from_end": best_frame_idx,
            "detection_method": detection_method,
            "all_detections": detection_results,
            "detection_settings": {
                "min_face_size": min_face_size,
                "eye_open_threshold": eye_open_threshold,
                "face_confidence": face_confidence,
                "require_frontal": require_frontal,
                "prefer_center": prefer_center
            }
        }
        
        if debug_output and debug_frames:
            debug_overlay = self._compile_debug_frames(debug_frames)
        else:
            debug_overlay = best_frame.unsqueeze(0) if best_frame.dim() == 3 else best_frame
        
        frames_to_trim = best_frame_idx
        
        # Debug: Verify we're returning the correct frame
        if torch.equal(best_frame, images[-1]):
            debug_print(f"CONFIRMED: Returning actual last frame (images[-1])")
        else:
            debug_print(f"WARNING: Returning frame that is NOT images[-1]")
            # Find which frame index this actually is
            for idx in range(total_frames):
                if torch.equal(best_frame, images[idx]):
                    debug_print(f"Actually returning images[{idx}] (frame {idx+1} of {total_frames})")
                    break
        
        return (
            best_frame.unsqueeze(0) if best_frame.dim() == 3 else best_frame,
            frames_to_trim,
            json.dumps(metadata),
            debug_overlay
        )
    
    def _detect_face(self, frame, method, min_face_size, eye_threshold, confidence):
        if method == "mediapipe":
            return self._detect_with_mediapipe(frame, min_face_size, eye_threshold, confidence)
        elif method == "mediapipe_detailed":
            return self._detect_with_mediapipe_detailed(frame, min_face_size, eye_threshold, confidence)
        else:
            return self._detect_bbox_fallback(frame, min_face_size, confidence)
    
    def _detect_with_mediapipe(self, frame, min_face_size, eye_threshold, confidence):
        result = {
            "has_face": False,
            "eyes_open": False,
            "is_frontal": False,
            "score": 0.0,
            "method": "mediapipe",
            "face_bbox": None,
            "eye_aspect_ratio": 0.0,
            "landmarks": None,
            "num_faces": 0,
            "all_faces_eyes_open": False
        }
        
        if not self.has_mediapipe:
            debug_print("MediaPipe not available, using fallback")
            return self._detect_bbox_fallback(frame, min_face_size, confidence)
        
        try:
            cv2_frame = self._tensor_to_cv2(frame)
            rgb_frame = self._bgr_to_rgb(cv2_frame)
            debug_print(f"MediaPipe processing frame shape: {rgb_frame.shape}, confidence: {confidence}, min_face_size: {min_face_size}")
            
            with self.mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=confidence) as face_detection:
                
                face_results = face_detection.process(rgb_frame)
                
                if face_results.detections:
                    debug_print(f"Found {len(face_results.detections)} face detections")
                    valid_faces = []
                    all_eyes_open = True
                    
                    # Check all detected faces
                    for i, detection in enumerate(face_results.detections):
                        # Get the actual confidence score MediaPipe found
                        detection_score = detection.score[0] if detection.score else 0.0
                        debug_print(f"Face {i+1} detected with confidence: {detection_score:.3f} (threshold: {confidence})")
                        bbox = detection.location_data.relative_bounding_box
                        h, w = rgb_frame.shape[:2]
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        debug_print(f"Face bbox: x={x}, y={y}, w={width}, h={height}, min_size={min_face_size}")
                        
                        if min(width, height) >= min_face_size:
                            valid_faces.append(detection)
                            debug_print(f"Face accepted (size check passed)")
                            
                            # Check eyes for this face
                            face_region = rgb_frame[max(0, y):min(h, y+height), max(0, x):min(w, x+width)]
                            if face_region.size > 0:
                                eye_ratio = self._analyze_eyes_with_facemesh(face_region)
                                debug_print(f"Eye aspect ratio: {eye_ratio}, threshold: {eye_threshold}")
                                if eye_ratio <= eye_threshold:
                                    all_eyes_open = False
                        else:
                            debug_print(f"Face rejected (too small: {min(width, height)} < {min_face_size})")
                    
                    if valid_faces:
                        result["num_faces"] = len(valid_faces)
                        result["has_face"] = True
                        result["all_faces_eyes_open"] = all_eyes_open
                        result["eyes_open"] = all_eyes_open  # Use all faces result
                        
                        # Use first face for other metrics
                        first_face = valid_faces[0]
                        bbox = first_face.location_data.relative_bounding_box
                        h, w = rgb_frame.shape[:2]
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        result["face_bbox"] = (x, y, width, height)
                        result["score"] = first_face.score[0]
                        
                        aspect_ratio = width / height if height > 0 else 0
                        result["is_frontal"] = 0.7 <= aspect_ratio <= 1.3
                        
                        # Get eye ratio from first face
                        eye_ratio = self._analyze_eyes_with_facemesh(rgb_frame)
                        result["eye_aspect_ratio"] = eye_ratio
                        
                        debug_print(f"Face detection: {len(valid_faces)} faces, all_eyes_open={all_eyes_open}")
                else:
                    debug_print(f"No face detections found")
                        
        except Exception as e:
            print(f"ERROR: MediaPipe detection failed: {e}")
            
        return result
    
    def _detect_with_mediapipe_detailed(self, frame, min_face_size, eye_threshold, confidence):
        result = self._detect_with_mediapipe(frame, min_face_size, eye_threshold, confidence)
        
        if result["has_face"] and self.has_mediapipe:
            try:
                cv2_frame = self._tensor_to_cv2(frame)
                rgb_frame = self._bgr_to_rgb(cv2_frame)
                
                with self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=confidence) as face_mesh:
                    
                    mesh_results = face_mesh.process(rgb_frame)
                    
                    if mesh_results.multi_face_landmarks:
                        landmarks = mesh_results.multi_face_landmarks[0]
                        result["landmarks"] = landmarks
                        
                        left_ear = self._calculate_eye_aspect_ratio(landmarks, "left")
                        right_ear = self._calculate_eye_aspect_ratio(landmarks, "right")
                        avg_ear = (left_ear + right_ear) / 2
                        
                        result["eye_aspect_ratio"] = avg_ear
                        result["eyes_open"] = avg_ear > eye_threshold
                        result["left_ear"] = left_ear
                        result["right_ear"] = right_ear
                        
            except Exception as e:
                debug_print(f"Detailed analysis error: {e}")
        
        result["method"] = "mediapipe_detailed"
        return result
    
    def _analyze_eyes_with_facemesh(self, rgb_frame):
        try:
            with self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
                
                results = face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    left_eye_ratio = self._calculate_eye_aspect_ratio(landmarks, "left")
                    right_eye_ratio = self._calculate_eye_aspect_ratio(landmarks, "right")
                    return (left_eye_ratio + right_eye_ratio) / 2
                    
        except Exception as e:
            debug_print(f"Eye analysis error: {e}")
            
        return 0.15
    
    def _calculate_eye_aspect_ratio(self, landmarks, eye_side):
        if eye_side == "left":
            eye_landmarks = [362, 380, 374, 263, 386, 385]
        else:
            eye_landmarks = [33, 159, 158, 133, 153, 145]
        
        try:
            def get_coords(idx):
                landmark = landmarks.landmark[idx]
                return np.array([landmark.x, landmark.y])
            
            points = [get_coords(idx) for idx in eye_landmarks]
            
            A = np.linalg.norm(points[1] - points[5])
            B = np.linalg.norm(points[2] - points[4])
            C = np.linalg.norm(points[0] - points[3])
            
            if C > 0:
                return (A + B) / (2.0 * C)
                
        except (IndexError, ZeroDivisionError, AttributeError) as e:
            debug_print(f"EAR calculation error for {eye_side} eye: {e}")
            
        return 0.15
    
    def _detect_bbox_fallback(self, frame, min_face_size, confidence):
        result = {
            "has_face": False,
            "eyes_open": True,
            "is_frontal": True,
            "score": 0.5,
            "method": "bbox_fallback",
            "face_bbox": None,
            "eye_aspect_ratio": 0.25
        }
        
        if isinstance(frame, torch.Tensor):
            h, w = frame.shape[-2:]
        else:
            h, w = frame.shape[:2]
        
        center_w = int(w * 0.4)
        center_h = int(h * 0.4)
        x = (w - center_w) // 2
        y = (h - center_h) // 2
        
        if min(center_w, center_h) >= min_face_size:
            result["has_face"] = True
            result["face_bbox"] = (x, y, center_w, center_h)
            result["score"] = confidence
        
        return result
    
    def _calculate_center_bonus(self, result, frame_shape):
        if not result["face_bbox"]:
            return 1.0
        
        x, y, w, h = result["face_bbox"]
        
        if isinstance(frame_shape, torch.Size):
            if len(frame_shape) >= 2:
                frame_h, frame_w = frame_shape[-2:]
            else:
                frame_h, frame_w = frame_shape
        else:
            frame_h, frame_w = frame_shape[:2]
        
        face_center_x = x + w / 2
        face_center_y = y + h / 2
        frame_center_x = frame_w / 2
        frame_center_y = frame_h / 2
        
        dist_x = abs(face_center_x - frame_center_x) / (frame_w / 2)
        dist_y = abs(face_center_y - frame_center_y) / (frame_h / 2)
        dist = np.sqrt(dist_x**2 + dist_y**2)
        
        return max(0.5, 1.0 - dist * 0.3)
    
    def _create_debug_overlay(self, frame, result):
        debug_frame = self._tensor_to_cv2(frame)
        debug_print(f"Creating debug overlay. Has bbox: {result['face_bbox'] is not None}, Has CV2: {self.has_cv2}")
        
        if result["face_bbox"] and self.has_cv2:
            x, y, w, h = result["face_bbox"]
            debug_print(f"Drawing bbox at ({x}, {y}, {w}, {h})")
            color = (0, 255, 0) if result["eyes_open"] else (0, 255, 255)
            
            self.cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, 2)
            
            status_lines = []
            if result["has_face"]:
                status_lines.append(f"Face: {result['score']:.2f}")
            if "eye_aspect_ratio" in result:
                status_lines.append(f"EAR: {result['eye_aspect_ratio']:.3f}")
            status_lines.append("Eyes: Open" if result["eyes_open"] else "Eyes: Closed")
            if result["is_frontal"]:
                status_lines.append("Frontal")
                
            for i, text in enumerate(status_lines):
                self.cv2.putText(debug_frame, text, (x, y - 10 - i * 20), 
                                self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return debug_frame
    
    def _compile_debug_frames(self, debug_frames):
        if not debug_frames:
            return None
        
        tensors = []
        for frame in debug_frames:
            if isinstance(frame, np.ndarray):
                frame_rgb = self._bgr_to_rgb(frame)
                tensor = torch.from_numpy(frame_rgb).float() / 255.0
            else:
                tensor = frame
            tensors.append(tensor)
        
        return torch.stack(tensors, dim=0)
    
    def _tensor_to_cv2(self, tensor):
        if tensor.dim() == 4:
            tensor = tensor[0]
            
        img = tensor.cpu().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
            
        img = (img * 255).astype(np.uint8)
        
        if self.has_cv2 and len(img.shape) == 3:
            img = self.cv2.cvtColor(img, self.cv2.COLOR_RGB2BGR)
            
        return img
    
    def _bgr_to_rgb(self, bgr_image):
        if self.has_cv2 and len(bgr_image.shape) == 3:
            return self.cv2.cvtColor(bgr_image, self.cv2.COLOR_BGR2RGB)
        return bgr_image


NODE_CLASS_MAPPINGS = {
    "FaceFrameDetector": FaceFrameDetector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceFrameDetector": "Face Frame Detector",
}