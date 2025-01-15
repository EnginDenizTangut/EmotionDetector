# KUTUPHANELER
import argparse
import sys
import time
from threading import Thread
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from collections import deque


mp_face_mesh = mp.solutions.face_mesh  
mp_drawing = mp.solutions.drawing_utils  
mp_drawing_styles = mp.solutions.drawing_styles  

COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None  

blendshape_scores = {}
blendshape_history = deque(maxlen=100)
time_stamps = deque(maxlen=100)
fig, ax = plt.subplots() #Butun degerler anlik olarak nasil degisiyor barplot grafiginde
fig2, ax2 = plt.subplots() #Sadece kullanilan degerler anlik olarak nasil degisiyor cizgi grafiginde

tracked_categories = [
    "left_smile", "right_smile", "browOuterUpLeft", "browOuterUpRight",
    "mouthPucker", "mouth_open", "smiling", "mouthShrugLower",
    "eyeLookDownLeft", "eyeLookDownRight", "jawOpen", "sleepy"
]#2.grafik icin gosterilmesi istenen yani kullanilan degerlerin isimlerini seciyoruz

emotion_counts = {
    "Smile": 0,
    "Sad": 0,
    "Surprised": 0,
    "Sleepy": 0
}
#her duygu icin counter baslangic degerlerini sifir olarak ayarliyoruz, counter degerleri ekran kac tane duygu ifadesini guncel olarak bize gostericek


#2.grafik icin guncel olarak degisen bir grafik olusturma
def update_time_series():
    """
    "Updates the changes in specific blendshape scores over time."
    """
    global blendshape_history, time_stamps

    ax2.clear()
    if len(blendshape_history) > 0:
        for key in tracked_categories:
            values = [frame.get(key, 0) for frame in blendshape_history]
            ax2.plot(time_stamps, values, label=key)
        """
        First, we start the process if there is any value inside the blendshape_history.
        Then, we sequentially take the values within tracked_categories.
        This fetches the relevant blendshape values for the key on each frame. If a frame contains the category specified in 'key', it adds the value to the values list; otherwise, it adds 0.
        Finally, the timestamps values are written on the x-axis, and the values values are written on the y-axis.
        """
        ax2.set_title("Tracked Blendshape Scores Over Time")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Score")
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper left')


def animate_time_series(i):
    update_time_series()

ani2 = FuncAnimation(fig2, animate_time_series, interval=100)
"""
FuncAnimation function creates an animation for a table with a dynamic variable. 
Plot the values of animate_time_series on the fig2 table."
"""

#1.grafik icin genel analiz sonuclarini gosteren grafik icin olusturulan fonksion
def update_graph():
    """
    Update blendshape scores
    """
    global blendshape_scores, DETECTION_RESULT
    ax.clear()  #grafigi temizler her cagrildiginda

    if DETECTION_RESULT and DETECTION_RESULT.face_blendshapes:
        face_blendshapes = DETECTION_RESULT.face_blendshapes[0]
        """
        Blendshape detects the face and converts it into 3D, allowing us to detect movements and perform operations on them.
        """
        for key in emotion_counts:
            emotion_counts[key] = 0
        #Her seferinde duyguları sıfırla

        # Her yüz için duygu tespiti
        for blendshapes in DETECTION_RESULT.face_blendshapes:
            left_smile = next((category.score for category in blendshapes if category.category_name == "mouthSmileLeft"), 0)
            right_smile = next((category.score for category in blendshapes if category.category_name == "mouthSmileRight"), 0)
            mouth_open = next((category.score for category in blendshapes if category.category_name == "mouthOpen"), 0)
            eye_look_down_left = next((category.score for category in blendshapes if category.category_name == "eyeLookDownLeft"), 0)
            eye_look_down_right = next((category.score for category in blendshapes if category.category_name == "eyeLookDownRight"), 0)
            mouth_shrug_lower = next((category.score for category in blendshapes if category.category_name == "mouthShrugLower"), 0)
            brow_outer_up_left = next((category.score for category in blendshapes if category.category_name == "browOuterUpLeft"), 0)
            brow_outer_up_right = next((category.score for category in blendshapes if category.category_name == "browOuterUpRight"), 0)
            jaw_open = next((category.score for category in blendshapes if category.category_name == "jawOpen"), 0)
            """
            "For example, for left_smile, it takes the value if available; if not, it assigns a value of 0 at that moment.            
            """
            # Smile
            if left_smile > 0.5 and right_smile > 0.5 and mouth_open < 0.5:
                emotion_counts["Smile"] += 1
            """
            For example, if the values of left_smile and right_smile are greater than 0.5, and mouth_open is less than 0.5, then increment the smile counter by 1 and display it on the screen."            
            """
            # Sad
            if eye_look_down_left > 0.45 and eye_look_down_right > 0.45 and mouth_shrug_lower > 0.5:
                emotion_counts["Sad"] += 1

            # Surprised
            if (brow_outer_up_left > 0.5 and brow_outer_up_right > 0.5) or (mouth_open > 0.5 and jaw_open > 0.5):
                emotion_counts["Surprised"] += 1

            # Sleepy
            if eye_look_down_left > 0.5 and eye_look_down_right > 0.5:
                emotion_counts["Sleepy"] += 1

        blendshape_scores = {
            category.category_name: category.score
            for category in face_blendshapes
        }
        """
        "We take the value for each emotion provided by the model, and then add it to the counter values."
        """

        categories = list(blendshape_scores.keys())  
        scores = list(blendshape_scores.values())  

        ax.barh(categories, scores, color='skyblue')
        ax.set_xlim(0, 1)  
        ax.set_title("Real-Time Blendshape Scores")
        ax.set_xlabel("Score")
        ax.set_ylabel("Blendshape Category")

def animate(i):
    update_graph()

ani = FuncAnimation(fig, animate, interval=100)  

def run(model: str, num_faces: int, min_face_detection_confidence: float,
        min_face_presence_confidence: float, min_tracking_confidence: float,
        camera_id: int, width: int, height: int) -> None:
    """
    Run the face detection and landmark tracking.
    """
    global FPS, COUNTER, START_TIME, DETECTION_RESULT

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    """
    "Here, we take the input camera, and then adjust the height and width settings coming from the camera."
    """

    def save_result(result: vision.FaceLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        """
        "Result contains the blendshape and landmark results of the detected faces."
        """
        global FPS, COUNTER, START_TIME, DETECTION_RESULT,blendshape_history,time_stamps

        # Her 10 karede bir fps hesapliyoruz burada
        if COUNTER % 10 == 0:
            FPS = 10 / (time.time() - START_TIME)
            START_TIME = time.time()
        """
        "For performance improvement, this value can be reduced to process more frames. 
        Another option is to decrease the camera resolution."
        """
        DETECTION_RESULT = result
        COUNTER += 1

        if result and result.face_blendshapes:
            face_blendshapes = result.face_blendshapes[0]
            current_scores = {
                category.category_name:category.score
                for category in face_blendshapes if category.category_name in tracked_categories
            }
            blendshape_history.append(current_scores)
            time_stamps.append(time.time())
        """
        "Our goal here is to monitor the model's performance by saving the detection results in each loop and making them accessible when needed."
        """


    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_faces=num_faces,
        min_face_detection_confidence=min_face_detection_confidence,
        min_face_presence_confidence=min_face_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_face_blendshapes=True,
        result_callback=save_result)
    detector = vision.FaceLandmarker.create_from_options(options)
    """
    base_options sets the basic options for the configuration of the model specified in the model_asset_path.  
    running_mode determines how the model operates. By selecting LIVE_STREAM, it indicates that the model will work in real-time video streaming.  
    num_faces defines the maximum number of faces to be detected.  
    min_face_detection_confidence sets the minimum required threshold for detecting a face.  
    min_face_presence_confidence sets the confidence threshold indicating that the face is truly present in the video frame.  
    min_tracking_confidence determines the required confidence level for tracking a face.  
    output_face_blendshapes enables the extraction of information about facial expressions. This is used to obtain the facial expression data.  
    result_callback stores the results when they are obtained, to be used elsewhere.  
    detector creates a face tracker (FaceLandmarker) object with the FaceLandmarkerOptions you defined earlier. This object will be used for tracking faces and detecting facial expressions.".
    """

    left_smile = False  
    right_smile = False
    eyeSquintLeft = False
    eyeSquintRight = False
    browOuterUpLeft = False
    browOuterUpRight = False
    mouthPucker = False
    mouth_open = False
    smiling = False
    mouthShrugLower = False
    eyeLookDownLeft = False
    eyeLookDownRight = False
    jawOpen = False
    eyeBlinkLeft = False
    eyeBlinkRight = False
    eyeLookDownLeft = False
    eyeLookDownRight = False
    upper_up_left = False
    upper_up_right = False
    """
    We assign flag values for all the parameters because when one is called, the other should not be called to avoid displaying multiple expressions on the screen simultaneously.
    The reason we set it to False is that we reset the values on each frame.
    In this line, we reset the values on each frame.
    The reason we define them outside is so that we can use these values in the same way after the loop."
    """

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)
        """
        cv2.flip inverts the image 180 degrees
        rgb_image converts the image from BGR to RGB, because cv2 works in this format
        image_format=mp.ImageFormat.SRGB: Specifies the image format. SRGB used here is a standard RGB color space. It indicates that the image data is in this color format.
        data=rgb_image: transfers image data to Mediapipe. rgb_image is an RGB image, usually converted from BGR with a library like OpenCV.
        detector.detect_async processes the image asynchronously
        """

        fps_text = f"FPS: {FPS:.1f}"
        cv2.putText(image, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        """
        Write the FPS value instantly in the upper left corner
        The reason why we do this is to be able to watch the performance tracking from the camera live.
        """
        # Sol üst köşede duyguları yazdır
        y_offset = 100  # İlk metin için başlangıç yüksekliği
        for emotion, count in emotion_counts.items():
            text = f"{emotion}: {count}"
            cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            y_offset += 30  # Bir sonraki metin için aşağı kaydır


        if DETECTION_RESULT and DETECTION_RESULT.face_landmarks:
            for landmarks in DETECTION_RESULT.face_landmarks:
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x,
                                                    y=landmark.y,
                                                    z=landmark.z) for landmark in landmarks
                ])
                """
                Here is the process of normalizing the points taken from the face, because we will show these values between 0 and 1 on the table
                The math behind the normalization process is as follows --> x_real / face_width
                Here is how face_width works: the model encloses the face in a bounding box, detects the extremes and then assigns values to face_weight, face_height
                Another reason why we normalize is actually to try to get every point roughly in the same place. Because everyone's face size is different in width and length.
                """
                mp_drawing.draw_landmarks(  
                    image=image,
                    landmark_list=face_landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,  
                    landmark_drawing_spec=None,  
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_tesselation_style())  
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_contours_style())
                """
                mp_drawing.draw_landmarks(), function to draw landmarks on the face
                image, image to process
                landmark_list allows us to extract and use a list of landmarks on the face
                connections connects points on the face to create a mesh network and attach it to the face
                landmark_drawing_spec, specifies the drawing properties (color, radius, thickness) of landmarks, (None default value)
                connection_drawing_spec, the drawing style of the connections, here it is supposed to be in a triangular structure
                connection_drawing_spec(2nd), Default settings of the contour drawing style are used for drawing connections.
                """
            if DETECTION_RESULT.face_blendshapes:
                blendshapes = DETECTION_RESULT.face_blendshapes[0]
                left_smile = False  
                right_smile = False
                eyeSquintLeft = False
                eyeSquintRight = False
                browOuterUpLeft = False
                browOuterUpRight = False
                mouthPucker = False
                jawOpen = False
                mouth_open = False
                smiling = False
                mouthShrugLower = False
                eyeLookDownLeft = False
                eyeLookDownRight = False
                eyeBlinkLeft = False
                eyeBlinkRight = False
                eyeLookDownLeft = False
                eyeLookDownRight = False
                upper_up_left = False
                upper_up_right = False
                """
                The values here are the result of calculations based on the mathematical movements of the facial expressions.
                We access these values in the graphical table that appears at first when the program runs.
                The main purpose of these, or more precisely the reason why we set them to False, is that these variables are reset to zero in each frame, so that when one emotion is perceived, the other one does not overtake it as much as possible.
                """

                for category in blendshapes:
                    #Sad Tespiti
                    """
                    if category.category_name == "eyeLookDownLeft" and category.score > 0.45:
                        eyeLookDownLeft = True
                    if category.category_name == "eyeLookDownRight" and category.score > 0.45:
                        eyeLookDownRight = True
                    """
                    """
                    Ilk if degerlerinde kontrol bu deger kamera uzerinden alindi mi ve bu deger 0.45 degerinden buyuk mu?
                    """
                    if category.category_name == "mouthShrugLower" and category.score > 0.5:
                        mouthShrugLower = True
                    if mouthShrugLower :
                        cv2.putText(image, "Sad", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (127, 10, 70), 2)
                        print("Sad")

                    """
                    --------------------------------------------------------------------------------------------------
                    # Landmark 33,133 ve 159: goz  (eyeLookDownLeft)
                    landmark_33 = (landmarks[33].x, landmarks[33].y, landmarks[33].z)
                    landmark_133 = (landmarks[133].x, landmarks[133].y, landmarks[133].z)
                    landmark_159 = (landmarks[159].x, landmarks[159].y, landmarks[159].z)

                    # Euclidean distance calculate function
                    def calculate_distance(p1, p2):
                        return np.linalg.norm(np.array(p1) - np.array(p2))

                    # distance calculate
                    distance_33_133 = calculate_distance(landmark_33, landmark_133)
                    distance_33_159 = calculate_distance(landmark_33, landmark_159)
                    distance_133_159 = calculate_distance(landmark_133, landmark_159)

                    --------------------------------------------------------------------------------------------------
                    # Sağ göz için Landmark noktaları(eyeLookDownRight)
                    landmark_33_right = (landmarks[33].x, landmarks[33].y, landmarks[33].z)
                    landmark_362 = (landmarks[362].x, landmarks[362].y, landmarks[362].z)
                    landmark_263 = (landmarks[263].x, landmarks[263].y, landmarks[263].z)

                    # distance calculate function
                    def calculate_distance(p1, p2):
                        return np.linalg.norm(np.array(p1) - np.array(p2))

                    # distance calculate
                    distance_33_362 = calculate_distance(landmark_33_right, landmark_362)
                    distance_33_263 = calculate_distance(landmark_33_right, landmark_263)
                    distance_362_263 = calculate_distance(landmark_362, landmark_263)

                    print(f"Distance between landmark[33] and landmark[362] (Right Eye): {distance_33_362:.3f}")
                    print(f"Distance between landmark[33] and landmark[263] (Right Eye): {distance_33_263:.3f}")
                    print(f"Distance between landmark[362] and landmark[263] (Right Eye): {distance_362_263:.3f}")
                    
                    --------------------------------------------------------------------------------------------------
                    # MouthShrugLower için Landmark noktaları (mouthShrugLower)
                    landmark_58 = (landmarks[58].x, landmarks[58].y, landmarks[58].z)
                    landmark_308 = (landmarks[308].x, landmarks[308].y, landmarks[308].z)
                    landmark_0 = (landmarks[0].x, landmarks[0].y, landmarks[0].z)

                    # distance calculate function
                    def calculate_distance(p1, p2):
                        return np.linalg.norm(np.array(p1) - np.array(p2))

                    # distance calculate
                    distance_58_308 = calculate_distance(landmark_58, landmark_308)
                    distance_58_0 = calculate_distance(landmark_58, landmark_0)
                    distance_308_0 = calculate_distance(landmark_308, landmark_0)

                    print(f"Distance between landmark[58] and landmark[308] (Mouth Shrug Lower): {distance_58_308:.3f}")
                    print(f"Distance between landmark[58] and landmark[0] (Mouth Shrug Lower): {distance_58_0:.3f}")
                    print(f"Distance between landmark[308] and landmark[0] (Mouth Shrug Lower): {distance_308_0:.3f}")
                    --------------------------------------------------------------------------------------------------
                    """
                    """
                    # Smile detection
                    if category.category_name == "eyeSquintLeft" and category.score > 0.5:
                        eyeSquintLeft = True
                    if category.category_name == "eyeSquintRight" and category.score > 0.5:
                        eyeSquintRight = True
                    if category.category_name == "mouthOpen" and category.score > 0.5:
                        mouth_open = True
                    
                    """
                    if category.category_name == "mouthSmileLeft" and category.score > 0.5:
                        left_smile = True
                    if category.category_name == "mouthSmileRight" and category.score > 0.5:
                        right_smile = True
                    if category.category_name == "mouthOpen" and category.score > 0.5:
                        mouth_open = True
                    if category.category_name == "mouthUpperUpLeft" and category.score > 0.5:
                        upper_up_left = True
                    if category.category_name == "mouthUpperUpRight" and category.score > 0.5:
                        upper_up_right = True

                    if left_smile and right_smile and (upper_up_left and upper_up_right):
                        smiling = True
                        cv2.putText(image, "Smile", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        print("Smile")



                    """
                    --------------------------------------------------------------------------------------------------
                    # Landmark 61 ve 0: Sol köşe ve üst köşe arasındaki distance
                    mouthSmileLeft = np.linalg.norm([
                    landmarks.landmark[61].x - landmarks.landmark[0].x,
                    landmarks.landmark[61].y - landmarks.landmark[0].y
                    ])
                    --------------------------------------------------------------------------------------------------
                    # Landmark 291 ve 12: Sağ köşe ve üst köşe arasındaki distance
                    mouthSmileRight = np.linalg.norm([  
                        landmarks.landmark[291].x - landmarks.landmark[12].x,
                        landmarks.landmark[291].y - landmarks.landmark[12].y
                    ])
                    --------------------------------------------------------------------------------------------------
                    mouthOpen = np.linalg.norm([
                    landmarks.landmark[13].x - landmarks.landmark[14].x,
                    landmarks.landmark[13].y - landmarks.landmark[14].y
                    ])

                    np.linalg.norm(), koordinati verilen iki nokta arasindaki farki bulmamizi sagliyor.
                    """

                    # Surprised tespiti
                    if category.category_name == "browOuterUpLeft" and category.score > 0.5:
                        browOuterUpLeft = True
                    if category.category_name == "browOuterUpRight" and category.score > 0.5:
                        browOuterUpRight = True
                    if category.category_name == "mouthPucker" and category.score > 0.5:
                        mouthPucker = True
                    if category.category_name == 'jawOpen' and category.score > 0.5:
                        jawOpen = True
                    if browOuterUpLeft and browOuterUpRight:
                        cv2.putText(image, "Surprised", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        print("Surprised")
                    if mouthPucker and jawOpen:
                        cv2.putText(image, "Surprised", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        print("Surprised")

                    """
                    --------------------------------------------------------------------------------------------------
                    # BrowOuterUpLeft ve BrowOuterUpRight calculate
                    browOuterUpLeft = np.linalg.norm([
                        landmarks.landmark[70].x - landmarks.landmark[105].x,
                        landmarks.landmark[70].y - landmarks.landmark[105].y
                    ])
                    --------------------------------------------------------------------------------------------------
                    browOuterUpRight = np.linalg.norm([
                        landmarks.landmark[300].x - landmarks.landmark[334].x,
                        landmarks.landmark[300].y - landmarks.landmark[334].y
                    ])
                    --------------------------------------------------------------------------------------------------
                    # MouthPucker calculatesı
                    mouthPucker = np.linalg.norm([
                        landmarks.landmark[78].x - landmarks.landmark[82].x,
                        landmarks.landmark[78].y - landmarks.landmark[82].y
                    ])
                    --------------------------------------------------------------------------------------------------
                    # JawOpen calculatesı ve MouthPucker kontrolü
                    jawOpen = np.linalg.norm([
                        landmarks.landmark[17].x - landmarks.landmark[0].x,
                        landmarks.landmark[17].y - landmarks.landmark[0].y
                    ])
                    # JawOpen içerisinde MouthPucker calculatesı
                    if jawOpen:
                        mouthPucker = np.linalg.norm([
                            landmarks.landmark[78].x - landmarks.landmark[82].x,
                            landmarks.landmark[78].y - landmarks.landmark[82].y
                        ])
                    --------------------------------------------------------------------------------------------------
                    """
                    # Sleepy Tespiti
                    """
                    if category.category_name == "eyeSquintLeft" and category.score > 0.5:
                        eyeSquintLeft = True
                    if category.category_name == "eyeSquintRight" and category.score > 0.5:
                        eyeSquintRight = True
                    """
                    if category.category_name == "eyeBlinkLeft" and category.score > 0.5:
                        eyeBlinkLeft = True
                    if category.category_name == "eyeBlinkRight" and category.score > 0.5:
                        eyeBlinkRight = True
                    if category.category_name == "eyeLookDownLeft" and category.score > 0.5:
                        eyeLookDownLeft = True
                    if category.category_name == "eyeLookDownRight" and category.score > 0.5:
                        eyeLookDownRight = True
                    if eyeBlinkLeft and eyeBlinkRight and eyeLookDownRight and eyeLookDownLeft and eyeSquintLeft and eyeSquintRight:
                        cv2.putText(image, "Sleepy", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (151, 10, 80), 2)
                        print("Sleepy")
                    """
                    # EyeBlinkLeft calculatesı
                    if eyeBlinkLeft:
                        eyeBlinkLeftDist = np.linalg.norm([
                            landmarks.landmark[159].x - landmarks.landmark[145].x,
                            landmarks.landmark[159].y - landmarks.landmark[145].y
                        ])

                    # EyeBlinkRight calculatesı
                    if eyeBlinkRight:
                        eyeBlinkRightDist = np.linalg.norm([
                            landmarks.landmark[386].x - landmarks.landmark[374].x,
                            landmarks.landmark[386].y - landmarks.landmark[374].y
                        ])

                    # EyeLookDownLeft calculatesı
                    if eyeLookDownLeft:
                        eyeLookDownLeftDist = np.linalg.norm([
                            landmarks.landmark[159].x - landmarks.landmark[23].x,
                            landmarks.landmark[159].y - landmarks.landmark[23].y
                        ])

                    # EyeLookDownRight calculatesı
                    if eyeLookDownRight:
                        eyeLookDownRightDist = np.linalg.norm([
                            landmarks.landmark[386].x - landmarks.landmark[253].x,
                            landmarks.landmark[386].y - landmarks.landmark[253].y
                        ])
                    """

        cv2.imshow('Face Mesh', image)
        
        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

def run_with_graph(model: str, num_faces: int, min_face_detection_confidence: float,
                   min_face_presence_confidence: float, min_tracking_confidence: float,
                   camera_id: int, width: int, height: int) -> None:
    thread = Thread(target=run, args=(model, num_faces, min_face_detection_confidence,
                                       min_face_presence_confidence, min_tracking_confidence,
                                       camera_id, width, height))
    thread.start()

    plt.show(block=False)
    plt.figure(fig.number)  # İlk grafik (Anlık Blendshape Skorları)
    plt.figure(fig2.number)  # İkinci grafik (Zaman Serisi)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='face_landmarker.task')
    parser.add_argument('--numFaces', default=1, type=int)
    parser.add_argument('--minFaceDetectionConfidence', default=0.5, type=float)
    parser.add_argument('--minFacePresenceConfidence', default=0.5, type=float)
    parser.add_argument('--minTrackingConfidence', default=0.5, type=float)
    parser.add_argument('--camera', default=0, type=int)
    parser.add_argument('--width', default=640, type=int)
    parser.add_argument('--height', default=480, type=int)

    args = parser.parse_args()
    run_with_graph(args.model, args.numFaces, args.minFaceDetectionConfidence,
                   args.minFacePresenceConfidence, args.minTrackingConfidence,
                   args.camera, args.width, args.height)
"""
Here we have added an argument constructor inside the maint function.
Our main goal is to give variables dynamically to the program.
We give the default values as we want
"""
if __name__ == '__main__':
    main()

"""
HOCALARIN SORDUGU SORULAR
1)Basimizi egdigimizde neden hareketler etkilenmiyor?
Cevap --> Modelimiz 3D modeli 2D modele cevirerek onun uzerinden islemler yapar, ondan dolayi kafamizi egmemiz onemli degil. 
Bu nedenle, kafa hareketleri gibi 3D değişiklikler modelin sonuçlarını doğrudan etkilemez, çünkü model yalnızca 2D projeksiyon üzerinde çalışır.

2)Hazir degerler aslinda neleri temsil ediyor?
Cevap --> Aslinda bizim aldigimiz degerler matematiksel bir islemin sonucu biz onu yapmak yerine direkt sonucu aliyoruz.
Burada mesela size ornek vermek istersem mouthSmileLeft icin matematiksel islemler su sekilde ilerliyor ve tabloda normalized edilmis halde 0 ve 1 arasinda degerler ile degisiyor.
Ornek vermek gerekirse mouthSmileLeft islemine gelene kadar su asamalardan geciyoruz;

x1, y1 = landmarks[0]['x'], landmarks[0]['y']  # Sol ağız köşesi (ornek olarak)
x2, y2 = landmarks[61]['x'], landmarks[61]['y']  # Ağız üstü nokta (örnek olarak)

# Ağız sol köşe ile üst köşe arasındaki distance calculate (örnek)
distance = np.linalg.norm([x2 - x1, y2 - y1])

3)Degerler neden 0.5 uzerinde hesaplaniyor?
Cevap --> Cunku 0.5 dedigimiz deger olmak yada olmamak kavramlarini tam ortadan 2'ye boldugu icin kesinlige daha yakin olmasi acisindan bu deger secilmistir

Kodda kullanılan Mediapipe modeli nedir ve nasıl çalışır?
Mediapipe, yüz ifadelerini analiz etmek için 472 landmark noktasını gerçek zamanlı olarak tespit eden, 3D yüz modelleme tabanlı bir sistemdir.

Neden blendshape skorlarını kullanıyoruz?
Blendshape skorları, yüz ifadelerini sayısal olarak temsil ederek her bir duygunun tespitini daha kolay ve güvenilir hale getirir.

Kodda kullanılan "tracked_categories" ne anlama gelir?
Tracked_categories, analiz edilmek istenen yüz ifadelerini veya mimikleri temsil eder, bu ifadelerin zaman içindeki değişimi grafikte gösterilir.

FPS değeri neden önemlidir?
FPS (Frame Per Second), modelin gerçek zamanlı performansını belirler ve kullanıcının deneyimini etkiler.

Mediapipe neden 3D noktaları normalize ediyor?
Normalize işlemi, farklı yüz boyutları ve pozisyonlarına rağmen yüz ifadelerinin sabit şekilde analiz edilmesini sağlar.

Kafa eğme hareketinde neden sonuçlar etkilenmiyor?
Model, 3D koordinatları analiz eder ve 2D projeksiyona dönüştürerek kafa hareketlerinden bağımsız çalışır.

Grafiklerle neyi göstermeyi hedefliyoruz?
Blendshape skorlarının ve yüz ifadelerinin zaman içindeki değişimlerini görselleştiriyoruz.

Eşik değeri olarak neden 0.5 kullanılıyor?
0.5, bir ifadenin varlığını güvenilir şekilde temsil etmek için seçilmiş bir eşik değeridir.

"Smile", "Sad", "Surprised" ifadeleri nasıl tespit ediliyor?
İlgili blendshape skorlarının belirli eşik değerlerini geçip geçmediğine bakılarak tespit ediliyor.

Modelin performansını nasıl değerlendiriyorsunuz?
Performans, blendshape skorlarının doğru ve hızlı şekilde tespit edilmesi ve FPS değeriyle ölçülür.



QUESTIONS ASKED BY TEACHERS
1) Why are the movements not affected when we tilt our head?
Answer --> Our model translates the 3D model into a 2D model and performs operations on it, so it doesn't matter if we tilt our head. 
Therefore, 3D changes such as head movements do not directly affect the results of the model, because the model only works on the 2D projection.

2) What do the ready-made values actually represent?
Answer --> Actually, the values we get are the result of a mathematical operation and instead of doing it, we get the result directly.
Here, for example, if I want to give you an example, the mathematical operations for mouthSmileLeft goes like this and it changes with values between 0 and 1, normalized in the table.
For example, we go through the following steps until we get to mouthSmileLeft;

x1, y1 = landmarks[0]['x'], landmarks[0]['y'] # Left mouth corner (for example)
x2, y2 = landmarks[61]['x'], landmarks[61]['y'] # Point on mouth (as an example)

# Calculate the distance between the left corner of the mouth and the top corner (example)
distance = np.linalg.norm([x2 - x1, y2 - y1])

3) Why are the values calculated at 0.5?
Answer --> Because 0.5 divides the concepts of being or not being exactly in the middle by 2, so this value is chosen to be closer to certainty.

What is the Mediapipe model used in the code and how does it work?
Mediapipe is a 3D face modeling based system that detects 472 landmark points in real-time to analyze facial expressions.

Why do we use blendshape scores?
Blendshape scores represent facial expressions numerically, making the detection of each emotion easier and more reliable.

What does “tracked_categories” mean in the code?
Tracked_categories represent the facial expressions or gestures to be analyzed, their change over time is shown in the graph.

Why is the FPS value important?
FPS (Frame Per Second) determines the real-time performance of the model and affects the user's experience.

Why does Mediapipe normalize 3D points?
Normalization ensures that facial expressions are analyzed consistently despite different face sizes and positions.

Why are the results unaffected by head tilt?
The model works independently of head movements by analyzing 3D coordinates and converting them into a 2D projection.

What do we aim to show with the graphs?
We visualize the changes in Blendshape scores and facial expressions over time.

Why use 0.5 as a threshold?
0.5 is a threshold value chosen to reliably represent the presence of an expression.

How are “Smile”, “Sad”, “Surprised” expressions detected?
They are detected by looking at whether the corresponding blendshape scores pass certain thresholds.

How do you evaluate the performance of the model?
Performance is measured by the accurate and fast detection of blendshape scores and the FPS value.

"""