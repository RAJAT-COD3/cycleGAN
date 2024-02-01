import streamlit as st
from cycleGAN_model import CycleGAN
import torch
from torchvision import transforms
from PIL import Image

# Load the trained model
model_path = "C:\Projects\cycleGAN\model\cycle_gan_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_state_dict = torch.load(model_path, map_location=device)

# Create an instance of your CycleGAN model and load the state_dict
loaded_model = CycleGAN(**MODEL_CONFIG)
loaded_model.load_state_dict(model_state_dict)
loaded_model.eval()
loaded_model.eval()

# Define the transformation for input images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Streamlit app
st.title("CycleGAN Image Translation App")

uploaded_file = st.file_uploader("Choose a photo...", type="jpg")

if uploaded_file is not None:
    # Preprocess the uploaded image
    input_image = Image.open(uploaded_file).convert("RGB")
    input_tensor = transform(input_image).unsqueeze(0)

    # Perform image translation using the loaded model
    with torch.no_grad():
        output_tensor = loaded_model.gen_PM(input_tensor.to(device))
    
    # Display the original and translated images
    st.image([input_image, transforms.ToPILImage()(output_tensor[0].cpu())], caption=['Original Image', 'Translated Image'], use_column_width=True)
