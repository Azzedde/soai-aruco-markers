from kivymd.app import MDApp
from kivymd.uix.button import MDRectangleFlatButton
from kivymd.uix.screen import Screen
from kivy.uix.image import Image
from kivy.uix.camera import Camera
from kivymd.uix.relativelayout import MDRelativeLayout
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import numpy as np
import arucoDetection as aruco


class DemoApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
    
    def build(self):
        screen = Screen()
        
        # Create a RelativeLayout
        rl = MDRelativeLayout()

        # Add the logo to the top right corner
        logo = Image(source='soai.jpg', size_hint=(0.2, 0.2), pos_hint={'top': 1, 'right': 0.9})
        rl.add_widget(logo)
        marker = Image(source='marker.png', size_hint=(0.12, 0.12), pos_hint={'top': 1, 'right': 0.3})
        rl.add_widget(marker)

        # Add a button to the bottom center
        button = MDRectangleFlatButton(text='Start', pos_hint={'center_x': 0.5, 'center_y': 0.15})
        button.bind(on_release=self.start_camera)
        rl.add_widget(button)

        # Add RelativeLayout to screen
        screen.add_widget(rl)
        
        return screen
    
    def start_camera(self, obj):
        augDicts = aruco.loadAugImages("ArucoImages")
        # Create camera object
        camera = Image( size_hint=(0.8, 0.6), pos_hint={'center_x': 0.5, 'center_y': 0.55})
        
        # Create a function to process the image 
        def process_image(dt):
            # Get the image from the capture 
            ret, img = self.capture.read()
            if not ret:
                print("Failed to grab frame")
                return
            # Process the image
            found = aruco.detectAruco(img)
            if len(found[0]) != 0:
                for bbox, id in zip(found[0], found[1]):
                    if int(id) in augDicts.keys():
                        img = aruco.generateAugmentedImage(bbox, id, img, augDicts[int(id)])
                        
            # display the image from the texture
            buf = cv2.flip(img, 0).tostring()
            texture1 = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            camera.texture = texture1
            # resolve
            



        
        # call the function each frame (1/60 sec)
        Clock.schedule_interval(process_image, 1.0 / 60.0)



        # Add camera widget to screen
        self.root.children[0].add_widget(camera)  # Indexing might change depending on your widget hierarchy


if __name__ == '__main__':
    DemoApp().run()
