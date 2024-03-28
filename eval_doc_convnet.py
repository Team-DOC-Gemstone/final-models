import keras
import numpy as np
from preprocessing import process_scan
model = keras.saving.load_model("3dconv_doc_lidc.keras", custom_objects=None, compile=True, safe_mode=True)
normal_raw_scan = "/fs/class-projects/spring2024/gems497/ge497g00/normal/001-patient-ge497g00.npy"
cancerous_path = "/fs/class-projects/spring2024/gems497/ge497g00/usable-cancerous/"
cancerous_raw_scan = cancerous_path + "4.000000-NA-00450-patient-0196.npy"
scan_norm = process_scan(cancerous_raw_scan)
prediction = model.predict(np.expand_dims(scan_norm, axis=0))[0]
scores = [1 - prediction[0], prediction[0]]

class_names = ["normal", "abnormal"]
for score, name in zip(scores, class_names):
    print("This model is %.2f percent confident that CT scan is %s"  % ((100 * score), name))
