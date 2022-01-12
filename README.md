# DiscoEditor
This is a Python 3 software to make a movide of a rotating disco and then, if required, syncronize an audio file with the video.
This idea was born from my necessity to create some funny videos for my dj sets, but I coudn't find any simple software on internet that can easily do this.
This script creates a cover with the given parameters, if you want you can stop the process to customize the cover yourself with other softwares (eg. Gimp).
When you're ready it takes the cover image and rotate it to create the rotating disco movie.

You can append a logo file with black or white backgroud, as the black logo in the example.
The software will replace the black or white color-space with custom color which you have choosen.


Example of the cover:

![Example of cover](https://github.com/Marco-Valle/DiscoEditor/blob/master/cover.png)

* You can see an example of video on Instagram:
https://www.instagram.com/tv/B-amAukJg2f/


### Prerequisites

1) Python 3
2) ffmpeg
3) pip3 (see requirequirements.txt)


### Installing

```
apt install python3
apt install ffmpeg
pip3 install -r requirements.txt 
```

## How it works

Modify the configuration file and then run the Python script as follow.

```
python3 editor.py cofiguration-file.txt OPTIONAL_FFMPEG_PATH
```

## Authors

* **Marco Valle** - [Marco-Valle](https://github.com/Marco-Valle)

## License

This project is licensed under the GPL v3 License - see the [LICENSE](LICENSE) file for details
