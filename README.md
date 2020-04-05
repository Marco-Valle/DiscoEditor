# DiscoEditor
This is Python 3 software to make a videos with rotating disco and then syncronize the audio file.
This idea was born from my necessity to create funny videos for my dj set, but I coudn't find any simple software on internet that could do this.
This script crate a cover with the given parameters, if you want you can stop the process to customize the cover yourself with other software (es. Gimp). When you're ready it takes the cover image and rotate it to create a rotating disco movie.

You could append a logo file with black or white backgroud as the black logo of example. The software'll replace black or white with custom disco color. See the configuration.py to all possible options.



Example of cover:

![Example of cover](https://github.com/Marco-Valle/DiscoEditor/blob/master/cover.png)

* You can see an example of video on Instagram:
https://www.instagram.com/tv/B-amAukJg2f/


### Prerequisites

1) Python 3
2) FFmpeg (better the version 4)
3) Pip3 (see requirequirements.txt)


### Installing

```
apt install python3
apt install ffmpeg
pip3 install -r requirements.txt 
```

## How it works

Modify the configuration-file.py and then run Python script.

```
python3 editor.py cofiguration-file
```

## Authors

* **Marco Valle** - [Marco-Valle](https://github.com/Marco-Valle)

## License

This project is licensed under the GPL v3 License - see the [LICENSE](LICENSE) file for details
