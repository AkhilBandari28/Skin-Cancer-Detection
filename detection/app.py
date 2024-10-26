from flask import Flask, request, render_template
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import torch
import timm


app = Flask(__name__)

# Define class names
class_names= {
    0: "Actinic_keratosis",
    1: "Basal_cell_carcinoma",
    2: "Benign_keratosis",
    3: "Dermatofibroma",
    4: "Melanoma",
    5: "Nevus",
    6: "Vascular_lesion",
}



model = timm.create_model('efficientformer_l1', pretrained=True)
model.load_state_dict(torch.load('checkpoints01/best_model_epoch_3.pt', map_location=torch.device('cpu')))
model.eval()

# Define image transformations
test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to process image and make predictions
# Function to process image and make predictions
def process_image(image):
    input_image = test_transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_image)

    predicted_class = torch.argmax(output, dim=1).item()
    confidence = F.softmax(output, dim=1)[0][predicted_class].item()
    predicted_class_name = class_names[predicted_class]
    cure_info = get_cure_info(predicted_class_name)

    return predicted_class_name, confidence, cure_info, image


# Function to encode image as base64
def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_img = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return encoded_img

def get_cure_info(predicted_class_name):
    # Define cure information
    cure_info = {
        "Actinic_keratosis": "Regular monitoring: Individuals with actinic keratosis should have regular skin examinations by a dermatologist to monitor for any changes or development of skin cancer. Sun protection: Using sunscreen with high SPF, wearing protective clothing, and avoiding sun exposure during peak hours can help prevent further skin damage. Topical treatments: Medications such as 5-fluorouracil, imiquimod, or ingenol mebutate may be prescribed to treat actinic keratosis. Cryotherapy: Freezing the lesions with liquid nitrogen is a common and effective treatment. Photodynamic therapy: This involves applying a photosensitizing agent to the skin and then exposing it to a specific wavelength of light to destroy abnormal cells.",

        "Basal_cell_carcinoma": "Surgical excision: Removing the cancerous tissue along with a margin of healthy skin is the most common treatment. Mohs surgery: A precise surgical technique that removes the cancer layer by layer, examining each layer under a microscope until no abnormal cells remain. Cryotherapy: Freezing the cancer cells with liquid nitrogen can be used for superficial BCC. Topical treatments: Medications such as imiquimod or 5-fluorouracil may be applied to the skin for superficial BCC. Radiation therapy: This may be used for patients who cannot undergo surgery. Regular follow-up: Patients treated for BCC should have regular skin exams to monitor for recurrence or new skin cancers.",

        "Benign_keratosis": "Regular monitoring: Dermatologists should monitor benign keratosis for any changes that might indicate malignancy. Topical treatments: Creams containing alpha hydroxy acids, urea, or salicylic acid can help reduce the appearance of benign keratosis. Cryotherapy: Freezing the lesion with liquid nitrogen may be used if it becomes symptomatic or cosmetically bothersome. Electrocautery: Burning the lesion off with electric current can be an effective treatment. Avoidance of irritants: Keeping the skin moisturized and avoiding irritants that may worsen the condition.",

        "Dermatofibroma": "Regular monitoring: Dermatofibromas are generally benign and do not require treatment unless they change in appearance or become symptomatic. Surgical excision: If the lesion is painful, itchy, or cosmetically concerning, it can be surgically removed. Cryotherapy: Freezing the lesion with liquid nitrogen can be an option, although it may not be as effective for deeper dermatofibromas. Topical treatments: Steroid creams may be used to reduce itching or inflammation.",

        "Melanoma": "Early detection: Regular skin checks and prompt evaluation of new or changing moles are crucial for early detection. Surgical excision: The primary treatment for melanoma is surgical removal of the tumor and a margin of surrounding healthy tissue. Sentinel lymph node biopsy: This may be performed to check for the spread of melanoma to nearby lymph nodes. Immunotherapy: Drugs that stimulate the immune system to attack melanoma cells may be used for advanced melanoma. Targeted therapy: Medications that target specific genetic mutations in melanoma cells can be effective. Radiation therapy: This may be used to treat melanoma that has spread to other parts of the body. Regular follow-up: Patients treated for melanoma need regular follow-up appointments to monitor for recurrence or new melanomas.",

        "Nevus": "Regular monitoring: Moles (nevi) should be monitored for any changes in size, shape, color, or symptoms such as itching or bleeding. Sun protection: Using sunscreen and wearing protective clothing can help prevent new moles and protect existing ones. Surgical excision: If a nevus shows signs of becoming atypical or malignant, it may be surgically removed. Biopsy: A suspicious nevus may be biopsied to determine if it is benign or malignant.",

        "Vascular_lesion": "Observation: Many vascular lesions do not require treatment and can be monitored for any changes. Laser therapy: Vascular lesions such as hemangiomas or port-wine stains can be treated with laser therapy to reduce their appearance. Surgical excision: If the lesion is symptomatic or cosmetically concerning, it may be surgically removed. Sclerotherapy: This involves injecting a solution into the lesion to shrink it. Regular follow-up: Regular skin exams are important to monitor the lesion and check for any changes."
    }

    return cure_info.get(predicted_class_name, "No cure information available.")




# Define Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file:
            image = Image.open(file).convert('RGB')
            predicted_class, confidence, cure_info, original_image = process_image(image)
            encoded_image = encode_image(original_image)

            return render_template('index.html', predicted_class=predicted_class, confidence=confidence, cure_info=cure_info, encoded_image=encoded_image)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
