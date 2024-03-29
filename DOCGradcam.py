

import numpy as np
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from preprocessing import process_scan
import cv2
# Load the model
model = tf.keras.models.load_model(r"DOC\final-models\run_3\3d_image_classification.tf")


# Load the input data
# input_data = np.load('30083.000000-NA-95222-patient-0075.npy')

input_data = process_scan('30083.000000-NA-95222-patient-0075.npy')


def visualize_slice(volume, slice_index):
    """Visualize a slice of the volume"""
    plt.imshow(volume[:, :, slice_index], cmap='gray')  # Assuming grayscale volume
    plt.axis('off')
    plt.title('Input Slice')
    plt.savefig("imgs_out/input.png")
    plt.show()


# visualize_slice(input_data, slice_index=12)
# exit()
# print(input_data)
# print(type(input_data))
# exit()
# Preprocess the input data
# (Apply necessary preprocessing steps like normalization or resizing)

# Convert input data to appropriate shape and type for the model
input_data = np.expand_dims(input_data, axis=0)  # Assuming batch size of 1
input_data = input_data.astype(np.float32)  # Ensure data type matches model's input data type
print(input_data.shape)
# Define a function to compute Grad-CAM
def compute_gradcam(model, img_array, layer_name):
    grad_model = Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]

 
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

    weights = tf.reduce_mean(guided_grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
    cam = np.maximum(cam, 0)  # ReLU
    cam /= np.max(cam)  # Normalize

    return cam

# Choose the layer for which you want to visualize activations
layer_name = 'conv3d_3'  # Example layer name from your model summary
# layer_name = 'max_pooling3d_3'  # Example layer name from your model summary

# Compute Grad-CAM
cam = compute_gradcam(model, input_data, layer_name)

plt.imshow(cam)
plt.show()
plt.clf()
plt.cla()
# exit()

    
def overlay_heatmap(input_image, heatmap, save_path):
    """Overlay the heatmap onto the input image and save the result"""
    print(heatmap.shape)
    print(input_image.shape)
    plt.imshow(input_image, cmap='gray')  # Display the input image

        # Resize heatmap to match input image dimensions
    resized_heatmap = np.zeros((256, 256, 4))
    for i in range(heatmap.shape[-1]):
        resized_heatmap[:, :, i] = cv2.resize(heatmap[:, :, i], (256, 256))

    plt.imshow(resized_heatmap, alpha=0.9, cmap='jet')  # Overlay the heatmap with transparency
    # plt.axis('off')
    plt.title('Input Image with Grad-CAM')
    plt.colorbar()
    plt.savefig(save_path)
    plt.clf()
    # exit()




input_image = input_data.reshape((256, 256, 1, 64))
print(input_data.shape)
for i in range(len(cam)):
    overlay_heatmap(input_image[:, :, 0, i*2], cam, f"imgs_out2/overlay_{i}.png")
    
    
# plt.save("imgs_out/out20.png")
# plt.show()

# def overlay_heatmap(input_image, heatmap, save_path):
#     """Overlay the heatmap onto the input image, scale heatmap, and save the result"""
#     scaled_heatmap = cv2.resize(heatmap, (input_image.shape[1], input_image.shape[0]))
#     # heatmap_normalized = cv2.normalize(scaled_heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

#     scaled_heatmap = np.uint8(255 * scaled_heatmap)

#     # Apply colormap
#     heatmap_colored = cv2.applyColorMap(scaled_heatmap, cv2.COLORMAP_JET)

#     # Overlay the heatmap with transparency
#     # overlaid_img = cv2.addWeighted(input_image, 0.5, heatmap_colored, 0.5, 0)
#     # input_image_3channel = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
#     # overlaid_img = heatmap_colored * .4 + input_image_3channel
#     # overlaid_img = input_image_3channel

#     # Save the result
#     cv2.imwrite(save_path, input_image)
    # cv2.imwrite(save_path, overlaid_img)


    # Scale up the heatmap to match the input image size (assuming input image is 256x256)

    
    # scaled_heatmap = cv2.resize(heatmap, (input_image.shape[1], input_image.shape[0]))

    # scaled_heatmap = cv2.normalize(scaled_heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # scaled_heatmap = np.uint8(255 * scaled_heatmap)  # Scale values back to [0, 255] range

    # # Apply colormap
    # heatmap_colored = cv2.applyColorMap(scaled_heatmap, cv2.COLORMAP_JET)

    # input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    # print(type(heatmap_colored), heatmap_colored.shape)
    # print(type(input_image_gray), input_image_gray.shape)

    # # Overlay the heatmap with transparency
    # # overlaid_img = cv2.addWeighted(input_image_gray, 0.5, heatmap_colored, 0.5, 0)
    # overlaid_img = cv2.addWeighted(cv2.cvtColor(input_image_gray, cv2.COLOR_GRAY2BGR), 0.5, heatmap_colored, 0.5, 0)

    # Save the result
    # cv2.imwrite(save_path, overlaid_img)


# for i in range(len(cam)):
#     overlay_heatmap(input_data[:, :, i, 0], cam[i], f"imgs_out/scaled_{i}.png")