This package aims at classifying two scan planes and labelling parts of the brain


---------INSTALLATION PROCEDURE------

-Install python
run: py -m pip install setuptools
run: "py setup.py install" in the directory where setup.py is found 


Install dependencies: Pillow, pytorch, numpy

py -m pip install numpy
py -m pip install pillow
py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


---------USAGE------

Run the "run.gui.bat" file to start the application

______-----PUSHING TO GIT WITH LARGE MODEL FILE-----
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add BrainImageAnalyzer_Pytorch/brainNet/Plane_detector_model.pth
git commit -m "Add large model file with Git LFS"
git push origin main



