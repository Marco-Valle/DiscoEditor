#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
This is an example of the configuration module of disco editor

'''

# Output file name (mp4 or avi) (don't use output.ext, the programme use it for temp file)
filename = 'out.mp4'

# Cover picture filename
cover_filename = 'cover.png'

# Logo filename (use None to no logo)
logo_filename = 'logo_black.png'

# Audio filename (use None to no audio)
music_filename = 'audio.mp3'

# Duration and format of video
seconds = 120
format_size = (1080,1920)

# Fps of video and the round per minute
fps=60
rpm=15

# Colors of background and of disco (BGR)
background_BGR = (78, 155, 0)
disco_BGR= (66, 245, 173)

# The logo has black background
logo_black_bg = True

# Tolerance during color replacing
logo_tolerance = 20

# Minimum padding of disco
min_padding = 50

# Rotation of disco
clockwise_rotation = True

# If you want interupt the program to customize the disco
custom_cover = False


