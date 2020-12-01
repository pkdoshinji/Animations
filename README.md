# Double_elastic

### Overview
Python-based animated simulation of the double elastic pendulum.

![](Images/octopus2.gif)

### Installation

```
git clone https://github.com/pkdoshinji/Animations
pip3 install numpy
pip3 install matplotlib
pip3 install scipy matplotlib
```

### Use
First create a subdirectory /frames in the directory where you are running the script.

Then, simply run:
```
./doubleelastic.py
```

To create a GIF, you will need ImageMagick.

Navigate to the frames directory and enter the following on the command line:
```
convert _img*.png <moviename.gif>
```

The GIF can be opened and displayed from your browser.
