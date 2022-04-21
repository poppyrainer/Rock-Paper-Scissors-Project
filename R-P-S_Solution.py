import random
import time
import cv2
from keras.models import load_model
import numpy as np

# This function uses the users web cam to capture an image, pass this to our keras model
# The model interprets the data and returns a probability of the result being rock/paper/scissors/nothing
# Note - the model was created and trained via Teachable Machines, are there are limitations with it, as it
# has only be trained in one light setting, with one person. It requires more training to provide an accurate result
def get_user_input():
    model = load_model('keras_model.h5')
    cap = cv2.VideoCapture(0)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    
    start_time = time.time()
    seconds = 3

    while True: 
        #create a timer for the webcam 
        current_time = time.time()
        elapsed_time = current_time - start_time

        #Open users webcam, capture data and pass to model
        print("Play!")
        ret, frame = cap.read()
        resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
        image_np = np.array(resized_frame)
        normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
        data[0] = normalized_image
        prediction = model.predict(data)
        cv2.imshow('frame', frame)

        #Convert model result into human readable result
        model_list = ['rock', 'paper', 'scissors','nothing']
        index_max = max(range(len(prediction)), key=prediction.__getitem__)
        human_prediction = model_list[index_max]

        if elapsed_time > seconds:
            break
                
    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    return human_prediction

# This function plays the game of rock paper scissors. It includes who wins each round
# and has a runningn total score. It follows the rule of "best of three rounds". So, if
# someone wins 2 rounds, they win the game 
def play_game():
    computer_score = 0
    user_score = 0 
    while computer_score <2 and user_score <2:

        #Generating computer choice randomly
        options_list=['rock', 'paper', 'scissors']
        computer_choice = random.choice(options_list)

        #Generating user choice via get_user_input function (i.e. the webcam and keras model)
        user_choice = get_user_input()
        #Error handling
        if user_choice == "nothing":
            print("no hand symbol recognised")
            continue

        print('computer choice: ' + computer_choice)
        print('user choice: ' + user_choice)

        #Working out the winner of each round
        if computer_choice == user_choice:
            result = "draw" 

        elif computer_choice == 'rock':
            if user_choice == 'paper':
                result = "user wins"
                user_score += 1
            if user_choice == 'scissors':
                result = "computer wins"
                computer_score += 1

        elif computer_choice == 'scissors':
            if user_choice == 'rock':
                result = "user wins"
                user_score += 1
            if user_choice == 'paper':
                result = "computer wins"
                computer_score += 1

        elif computer_choice == 'paper':
            if user_choice == 'rock':
                result = "computer wins"
                computer_score += 1
            if user_choice == 'scissors':
                result = "user wins"
                user_score += 1

        print(result)
        print("user score: " + str(user_score))
        print("computer score: " + str(computer_score))
    
    #Calculating the overall winner
    if user_score > computer_score:
        final_result = "user wins!"
    else:
        final_result = "computer wins!"
    
    print("the final result is: " + final_result)

play_game()
