# Driver-Drowsiness-Detection
Driver drowsiness detection is a Sem 5 mini project for my computer engineering college built using Dlib and OpenCV with Python as a backend language.

---

# Logic of project
The project includes direct working with the 68 facial landmark detector and also the face detector of the Dlib library.
The 68 facial landmark detector is a robustly trained efficient detector which detects the points on the human face using which 
we determine whether the eyes are open or they are closed.</br></br>
<p align="center">
  <img src="images/landmarks.jpg" height="350">
</p>

# Important files to download
Please download these files before running the project:

<b>1) Python version required: 3.10.</b><b> <a href="https://www.python.org/downloads/release/python-3100/">It can be found here</a></b>. 

**Imp Note:** Pls add it to PATH on System variables by clicking on PATH then Edit and then New then press Ok to save.

<b>2) The 68-landmark detector data (.dat) file can be found <a href="https://github.com/davisking/dlib-models">by clicking here</a></b>.

<b>3) Dlib Wheel file <a href="https://github.com/z-mahmud22/Dlib_Windows_Python3.x/blob/main/dlib-19.22.99-cp310-cp310-win_amd64.whl">can be found here</a></b>

---

# Virtual environment command:
1. Activation
   ```bash
   py -3.10 -m venv venv
   venv\Scripts\activate

2. Deactivation
   ```bash
   deactivate

---

## To install the libraries
1. Here is the command
   ```bash
   pip install -r requirements.txt 

# The working of the project
<ul><li>As you can see the<b> above screenshot</b> where the landmarks are detected using the detector.
<li>Now we are taking the ratio which is described as <i>'Sum of distances of vertical landmarks divided by twice the distance between horizontal landmarks'</i>.
<li>Now this ratio is totally dependent on your system which you may configure accordingly for the thresholds of sleeping, drowsy, active.</ul>
<p align="center">
  <img src="images/active.jpg" height="350">
  <br></br>
  <img src="images/sleepy.jpg" height="350">
  <br></br>
  <img src="images/drowsy.jpg" height="350">
</p>

---

# How to Download the Project?
1) Click on the Code option.
2) Then Select 'Download Zip'.
3) Extract the folder.
4) Install all the important files in [Important files to download](#important-files-to-download) section and the [To install the libraries](#to-install-the-libraries) section. Both of these files in the sections are *very important* to run the project.
5) Use python version 10 for best results.
6) Please run the **driver.py** file in Visual Studio Code. (It works)

---


## 👩‍💻 Contributors/Collaborators  
- [Saileen Fernandes (@Sai25Hajime)](https://github.com/Sai25Hajime)
- [Jelestina Nadar (@Jelestina)](https://github.com/Jelestina)
- [Alister Almeida (@miali4657)](https://github.com/miali4657)


