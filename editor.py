#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import imutils
import importlib
from progress.bar import Bar
from os import remove, rename, system


class picture:

    def __init__(self, filename='example.png'):
        self.filename = filename
        self.img = None

    def __repr__(self):
        return 'Picture()'

    def __str__(self):
        if self.img is None:    return "No such file or directory"
        else:   return "Picture Ojbect\nFilename: {}\nShape: {}".format(self.filename, (self.height, self.width))
        
    def import_img(self, force_import=False):
        if self.img is None or force_import:    self.img = cv2.imread(self.filename) # If there's not image set try to import
        if self.img is None:    raise IOError("No such file or directory") # If opencv caon't find it raise error
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        (self.height, self.width) = self.gray.shape

    def next(self):
        # This function allow to change the current image with the prevoius resulting image
        self.img = self.result
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        (self.height, self.width) = self.gray.shape

    def resize(self, new_shape):
        self.result = cv2.resize(self.img, new_shape)
        self.next() # Update object values

    def crop(self, center_coordinates=None, width=None):
        img = self.img
        if center_coordinates and width:
            x = center_coordinates[0]-width
            y = center_coordinates[1]-width
            w = 2*width
            h = w
        else:
            (ret, thresh) = cv2.threshold(self.gray, 50, 255, cv2.THRESH_BINARY)
            # Create mask
            mask = np.zeros((self.height, self.width), np.uint8)
            edges = cv2.Canny(thresh, 100, 200)
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 10000, param1=50, param2=30, minRadius=0, maxRadius=0)
            for i in circles[0, :]:
                i[2] = i[2] + 4
                cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), thickness=-1) # Draw on mask
            # Apply Threshold
            (_, thresh) = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            # Find Contour
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            (x, y, w, h) = cv2.boundingRect(contours[0][0])
        # Crop masked_data
        self.result = img[y:y + h, x:x + w]
    
    def rotate_bound(self, angle, BGR = None):
        ## IMUTILS: redefine this function
        image = self.img
        # Grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # Grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # Compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # Adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        # Perform the actual rotation and return the image
        # Custom: I append the possibility to cahnge the background color fix
        self.result = cv2.warpAffine(image, M, (nW,nH), borderMode=cv2.BORDER_CONSTANT, borderValue=BGR) if BGR else cv2.warpAffine(image, M, (nW, nH))

    def rotate_IMUTILS(self, angle, BGR = None, center=None, scale=1.0):
        ## IMUTILS: redefine this function
        image = self.img
         # Grab the dimensions of the image
        (h, w) = image.shape[:2]
        # If the center is None, initialize it as the center of
        # the image
        if center is None:  center = (w // 2, h // 2)
        # Perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        self.result = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=BGR) if BGR else cv2.warpAffine(image, M, (w, h)) # Custom

    def rotate(self, min=0, max=360, step=15, get_array=False, fix=True, BGR_fix = None):
        if get_array:   self.results = []
        for angle in np.arange(min, max, step):
            if fix: # Fix the shape of picture (it's usefull when you haven't square picture)
                if BGR_fix: # This's the background fix
                    try:    self.rotate_bound(angle, BGR=BGR_fix)
                    except Exception as e:  print("Exception: {}".format(e))
                else:   self.result = imutils.rotate_bound(self.img, angle) # If custom function doesn't work it tries with IMUTILS original function
            else:
                if BGR_fix:
                    try:    self.rotate_IMUTILS(angle, BGR=BGR_fix)
                    except Exception as e:  print("Exception: {}".format(e))       
                else:   self.result = imutils.rotate(self.img, angle)
            if get_array:   self.results.append(self.result)

    def show(self, img=None, img_array=None, label='Label'):
        if img_array is None:
            if img is None: img = self.img
            img_array = [img] # If we have single image it creates one element array
        for i in img_array:
            cv2.imshow(label, i)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
        if len(img_array) == 1: return key # If we have single image we return the key pressed

    def save_img(self, filename='img.png', img=None):
        if img is None:   cv2.imwrite(filename, self.img)
        else:   cv2.imwrite(filename, img)


class video:

    def __init__(self, frames, format_size = (1024,640), loops=1, fps=30, background_BGR = (255,255,255), clockwise_rotation = False, music_filename = None):
        self.frames = frames # Frames of video
        self.format_size = format_size # Video Resolution
        self.background_BGR = background_BGR
        self.fps = fps
        self.loops = loops # Number of loops of disco rotation
        height, width, layers = self.frames[0].shape
        self.size = (width,height)
        self.filename = "out.avi"
        self.music_filename = music_filename
        self.n = len(self.frames) # Number of frames
        self.seconds = (self.n*self.loops)/self.fps # Video's duration
        self.clockwise_rotation = clockwise_rotation

    def __repr__(self):
        return 'Video()'

    def __str__(self):
        return "\nVideo Ojbect\nFormat: {}\nAudio: {}\nFrames: {}\nTotal frames generated: {}\nShape of frames: {}\nEstimed video duration: {} seconds ({} minutes)\n".format(self.format_size, self.music_filename, self.n, self.n*self.loops, self.size, self.seconds, self.seconds/60)        

    def set_rotation(self):
        if self.clockwise_rotation:	self.frames.reverse()

    def add_padding(self, img=None):
        frames = []
        base_size=(self.format_size[0], self.format_size[1], 3) # Numpy shape is before the height and then width
        base=np.zeros(base_size, dtype=np.uint8)
        cv2.rectangle(base,(0,0),(self.format_size[1], self.format_size[0]), self.background_BGR, -1) #
        x0 = int(base_size[1]/2 - self.size[0]/2)
        x1 = int(base_size[1]/2 + self.size[0]/2)
        y0 = int(base_size[0]/2 - self.size[1]/2)
        y1 = int(base_size[0]/2 + self.size[1]/2)
        if img is not None: # If we need to add padding to single image
            base[y0:y1, x0:x1] = img
            return base
        print("\nAdd padding to frames:")
        # Create loading bar
        with Bar('Processing', max=len(self.frames)) as bar: 
            for frame in self.frames:
                tmp = base.copy()
                tmp[y0:y1, x0:x1] = frame
                frames.append(tmp)
                bar.next()
        self.frames = frames
        height, width, layers = self.frames[0].shape
        self.size = (width,height)
        print("Done")

    def add_audio(self):
        # Moviepy caused some problem, so I decided to use terminal based FFMPEG
        print("\n\n")
        ext = self.filename.split('.')[1]
        tmp_filename = 'output.{}'.format(ext)
        try:    remove(tmp_filename) # Check if there's any old file and remove it
        except: pass
        try:    system("ffmpeg -i {} -i {} -c copy -map 0:v:0 -map 1:a:0 -shortest {}".format(self.filename, self.music_filename, tmp_filename))
        except: print("Something is gone wrong during the audio syncronization")
        try:
            remove(self.filename)        
            rename(tmp_filename, self.filename)
        except: print("Something is gone wrong at the end, check if there's file names output.{}".format(ext))

    def create(self, filename="out.mp4"):
        # Check if frames shape and format size are the same
        if np.array_equal(self.size, self.format_size) is False:    self.add_padding()
        self.filename = filename
        # Select format
        try:    avi = True if filename.split('.')[1]=='avi' else False
        except: avi = False
        fourcc = cv2.VideoWriter_fourcc(*'XVID') if avi else cv2.VideoWriter_fourcc(*'FMP4')
        # Write on file
        out = cv2.VideoWriter(filename, fourcc, fps=self.fps, frameSize=self.size)
        print("\nWriting video:")
        # Create loading bar
        with Bar('Processing', max=self.loops) as bar: 
            for i in range(int(self.loops)):
                for frame in self.frames:   out.write(frame)    # Write each frame
                bar.next()
        out.release()
        if self.music_filename: self.add_audio()
        try:
            if self.filename.split('.')[1] == 'mp4': # Convert to H264 codec
                tmp_filename = 'output.mp4'
                try:    remove(tmp_filename) # Check if there's any old file and remove it
                except: pass
                system("ffmpeg -i {} -vcodec libx264 {}".format(self.filename, tmp_filename))
                try:
                    remove(self.filename)        
                    rename(tmp_filename, self.filename)
                except: print("Something is gone wrong at the end, check if there's file names output.mp4")
        except: print("Error during video conversion")
        print("\nVideo termined > {}".format(self.filename))


class editor:

    def __init__(self, configuration_module, filename='example.png'):
        # User module data
        try:
            self.filename = configuration_module.filename
            self.logo_filename = configuration_module.logo_filename # The filename of the logo to put on cover
            self.cover_filename = configuration_module.cover_filename
            self.music_filename = configuration_module.music_filename
            self.seconds = configuration_module.seconds
            self.fps = configuration_module.fps
            self.rpm = configuration_module.rpm
            # The background color (it needs this data to fix the background during the rotation)
            self.background_BGR = configuration_module.background_BGR
            self.disco_BGR = configuration_module.disco_BGR
            self.logo_black_bg = configuration_module.logo_black_bg
            self.logo_tolerance = configuration_module.logo_tolerance
            self.format_size = configuration_module.format_size
            min_padding = configuration_module.min_padding
            self.clockwise_rotation = configuration_module.clockwise_rotation
            # If you want iterupt the video creaton for the cover customization (with Gimp or Paint for examples)
            self.custom_cover = configuration_module.custom_cover
        except:	raise Exception("Something is gone wrong with configuration file")

        # Program data
        self.draw = True # Need to draw the disco (change only in developing)
        self.p = picture() # Picture object
        self.ready = False # If user is ready to create the video
        self.side = min(self.format_size) - min_padding*2

    def __repr__(self):
        return 'Editor()'

    def __str__(self):
        try:    return "\nEditor Ojbect\nSeconds: {}\nFps: {}\nFrames Required: {}\nStep: {}°\nLoops: {}\nRpm: {}\nFpr: {}".format(
                    self.seconds, self.fps, self.framesRequired, self.step, self.loops, self.rpm, self.fpr)
        except: return "\nEditor object: no data to comunicate\nBefore calculate params"
   
    def calculate_params(self):
        self.framesRequired = self.seconds * self.fps # Number of required frames for required duration
        rps = self.rpm/60 # Convert in round per second
        self.fpr = self.fps/rps # Frame per round [f/r]
        self.step = 360 / self.fpr # The angles step (es. step=60 -> in position [0,60,120,180,240,300,360] degrees disco'll be photographed)
        self.loops = self.framesRequired/self.fpr # Loops are given from simple division (frames required / frames required per round)
        # Loop must be integer
        if self.loops.is_integer() is False:    print("\n\nWarning: loops is float, it'll be aproximated.\nThis may cause some problems.\n\n") 

    def add_logo(self, black_background = False, disco_BGR=(255,255,255), shape=(500,500), tolerance=20):
        # Add logo in the center
        # Logo with black or white background is required
        self.logo = picture(self.logo_filename)
        self.logo.import_img()
        print("\nCurrent logo shape (x,y): {}".format((self.logo.width, self.logo.height)))
        report = self.logo.width/self.logo.height
        if self.logo.width>=self.logo.height: # It changes the larger mesure and keep report to calculate the other
            logo_shape_x = int(shape[0]-(shape[0]/20))
            logo_shape_y = int(logo_shape_x * 1/report)
        else:
            logo_shape_y = int(shape[1]-(shape[1]/20))
            logo_shape_x = int(logo_shape_y * report)
        logo_shape = (logo_shape_x, logo_shape_y)
        print("Resizing logo, new shape: {}".format(logo_shape))
        self.logo.resize(logo_shape) # Resize to disco's shape less 5%
        if isinstance(tolerance,int) is False: int(tolerance) # We need integer
        lower = np.array(0, dtype = "uint8") if black_background else np.array(255-tolerance, dtype = "uint8")
        upper = np.array(tolerance, dtype = "uint8") if black_background else np.array(255, dtype = "uint8")
	# Find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(self.logo.img, lower, upper)
        self.logo.img[mask>0] = disco_BGR # Replace the logo background with disco's color
        x0 = int(self.result.shape[1]/2 - self.logo.img.shape[1]/2)
        x1 = int(self.result.shape[1]/2 + self.logo.img.shape[1]/2)
        y0 = int(self.result.shape[0]/2 - self.logo.img.shape[0]/2)
        y1 = int(self.result.shape[0]/2 + self.logo.img.shape[0]/2)
        self.result[y0:y1, x0:x1] = self.logo.img

    def create_cover(self, disco_BGR=(255,255,255), side=1200, thickness=2, small_radius=0.08):
        background = np.full((side,side,3), self.background_BGR, dtype=np.uint8) # Draw colored background
        big_radius = int(side/2-thickness) # Big radius must be the half of the side
        # Calculate the number of pixels of the small radius
        small_radius = int(big_radius * small_radius) if (small_radius <=1 and small_radius >0) else int(big_radius * (1/small_radius)) 
        self.width = big_radius+thickness
        self.center_coordinates = (int(side/2), int(side/2))
        self.result = cv2.circle(background, self.center_coordinates, big_radius, (0,0,0), thickness) # Draw the disco's shape
        self.result = cv2.circle(self.result, self.center_coordinates, big_radius-1, disco_BGR, -1)  # Color the disco
        if self.logo_filename:
            self.add_logo(black_background=self.logo_black_bg, disco_BGR=disco_BGR, shape=(side,side), tolerance=self.logo_tolerance)
        self.result = cv2.circle(self.result, self.center_coordinates, small_radius, (0,0,0), thickness) # Draw the center hole
        self.result = cv2.circle(self.result, self.center_coordinates, small_radius-1, self.background_BGR, -1) # Color it with background color

    def suspend(self):
        # This function allow to user to make some manually modifications and then import the new image
        self.p.filename = 'tmp_cover.png'
        print("Saving the image of cover > {}\nWhen you're ready press Enter (be sure to have saved the file with same name)".format(self.p.filename))
        self.p.save_img(filename=self.p.filename)
        input("Ready?")
        self.p.import_img(force_import=True)
        try:    remove(self.p.filename)
        except: pass
        
    def run(self):
        print("FFMPEG is required (version 4 is better)")
        if self.draw:
            self.create_cover(disco_BGR=self.disco_BGR, side=self.side)
            self.p.img = self.result # Import in picture object the image generated
        self.calculate_params() # Set automatically params
        self.p.import_img() # Complete the image importation
        self.p.crop(center_coordinates=self.center_coordinates, width=self.width)
        self.p.next()
        if self.custom_cover:   self.suspend()
        cover = self.p.img
        self.p.rotate(step=self.step, get_array=True, fix=False, BGR_fix=self.background_BGR) # Create the frames eith rotating disco     
        print(self) # Get some infos
        self.v = video(
            self.p.results, fps=self.fps, loops=self.loops, background_BGR=self.background_BGR, format_size=self.format_size,
            clockwise_rotation=self.clockwise_rotation, music_filename=self.music_filename) # Video Object
        self.v.set_rotation()
        print("{}\nClick on picture and press c to confirm".format(self.v))
        self.v.frames.insert(0, cover) # Add cover at the video beginning
        cover = self.v.add_padding(img=cover)
        key = self.p.show(cover, label="Preview")
        if key == ord('c') and self.v.seconds != 0:  self.ready = True # User's ready
        else:   print("Discard")
        if self.ready is False: return False
        self.p.save_img(filename=self.cover_filename, img=cover)
        self.v.create(filename=self.filename)

        
if __name__ == '__main__':
    from sys import argv, exit
    from shutil import rmtree
    try: rmtree("__pycache__") # Remove cache to reload configuration file
    except: pass
    try:
        cnf = importlib.import_module(argv[1].replace('.py',''))
        e = editor(configuration_module=cnf)
    except:
        print("Usage: python3 editor.py configuration")
        exit(0)
    e.run()
