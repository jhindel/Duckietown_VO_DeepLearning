import onnxruntime as rt
import onnx

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

# TODO copy final config of model
args = {"resize": 64}

# TODO copy final config of model
# same as for training of model
transform = transforms.Compose([
    transforms.Resize((args["resize"], args["resize"])),
    transforms.ToTensor(),
])

# TODO copy final config of model
def preprocess_img(img):
    area = (0, 160, 640, 480)
    img = img.crop(area)
    img = transform(img)
    return img


# We have to stack two images before to push in the CNN
def preprocess_pose(input_data1, input_data2):
    # convert the 2 input data into float32 input
    img1 = Image.open(input_data1).convert('RGB')
    img2 = Image.open(input_data2).convert('RGB')
    img1 = preprocess_img(img1)
    img2 = preprocess_img(img2)
    newimg = np.concatenate([img1, img2], axis=0)
    newimg = newimg[np.newaxis, ...]
    return newimg


def display_images(img):
    fig, (ax0, ax1) = plt.subplots(1, 2)
    img1 = np.moveaxis(img[0, :3], 0, 2)
    img2 = np.moveaxis(img[0, 3:], 0, 2)
    ax0.imshow(np.asarray(img1))
    ax0.set_title('Image 1')
    ax1.imshow(np.asarray(img2))
    ax1.set_title('Image 2')
    fig.show()

def CNN_processing(img)
	image2=image1
	image1=img
	if not (image2):
	#TODO skip the first time if it's the first image to the input
		return #empty
	else:
		input_to_CNN = preprocess_pose(image1, image2)
		#display_images(input_to_CNN) #Optional if we would want to see the pics
		
		model_path = '/home/baptiste/Downloads/deepvo/solution/src/DeepVo/2021-12-27-16-43_bestmodel.onnx'
		onnx_model = onnx.load(model_path)

		# Check the model
		onnx.checker.check_model(onnx_model)

		# Run model
		session = rt.InferenceSession(model_path)
		outputs = session.run(None, {'input': input_to_CNN})

		return outputs #TODO Define the output values of ONNX file
	


#img1_name = '/Users/julia/Documents/UNI/Master/Montréal/AV/project/duckietown_visual_odometry/test_onnx/frame000065.png'
#img2_name = '/Users/julia/Documents/UNI/Master/Montréal/AV/project/duckietown_visual_odometry/test_onnx/frame000066.png'
#input_to_CNN = preprocess_pose(img1_name, img2_name)
#display_images(input_to_CNN)

# Run the CNN model
#model_path = '/Users/julia/Documents/UNI/Master/Montréal/AV/project/duckietown_visual_odometry/#test_onnx/2021-12-27-16-43_bestmodel.onnx'
#onnx_model = onnx.load(model_path)

# Check the model
#onnx.checker.check_model(onnx_model)

# Run model
#session = rt.InferenceSession(model_path)
#outputs = session.run(None, {'input': input_to_CNN})

#print(outputs)

