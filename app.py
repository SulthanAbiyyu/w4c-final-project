import streamlit as st
import torch
from PIL import Image
import torchvision
from torchvision.transforms import v2
from torchvision.models import mobilenet_v3_small
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
from torch.cuda.amp import autocast
import torchvision.transforms as transforms

model_path = "./model/train_acc_best.pt"

class_groups = {
    'Plastics': [
        'disposable_plastic_cutlery', 'plastic_cup_lids', 'plastic_detergent_bottles',
        'plastic_food_containers', 'plastic_shopping_bags', 'plastic_soda_bottles',
        'plastic_straws', 'plastic_trash_bags', 'plastic_water_bottles'
    ],
    'Metals': [
        'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 'steel_food_cans'
    ],
    'Glass': [
        'glass_beverage_bottles', 'glass_cosmetic_containers', 'glass_food_jars'
    ],
    'Cardboard/Paper': [
        'cardboard_boxes', 'cardboard_packaging', 'magazines', 'newspaper', 
        'office_paper', 'paper_cups'
    ],
    'Food Waste': [
        'coffee_grounds', 'eggshells', 'food_waste', 'tea_bags'
    ],
    'Textiles': [
        'clothing', 'shoes'
    ],
    'Styrofoam': [
        'styrofoam_cups', 'styrofoam_food_containers'
    ]
}

example_images = {
    "Plastic": "./img/example/plastic.png",
    "Metal": "./img/example/metal.png",
    "Glass": "./img/example/glass.png",
}

class_names = [
        'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans',
        'cardboard_boxes', 'cardboard_packaging', 'clothing',
        'coffee_grounds', 'disposable_plastic_cutlery', 'eggshells',
        'food_waste', 'glass_beverage_bottles', 'glass_cosmetic_containers',
        'glass_food_jars', 'magazines', 'newspaper', 'office_paper',
        'paper_cups', 'plastic_cup_lids', 'plastic_detergent_bottles',
        'plastic_food_containers', 'plastic_shopping_bags',
        'plastic_soda_bottles', 'plastic_straws', 'plastic_trash_bags',
        'plastic_water_bottles', 'shoes', 'steel_food_cans',
        'styrofoam_cups', 'styrofoam_food_containers', 'tea_bags'
    ]

st.set_page_config(page_title="Waste Thinker 1.0", page_icon="♻", layout="centered")

class WasteClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobnet = mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.feature_extraction = create_feature_extractor(self.mobnet, return_nodes={'features.12': 'mob_feature'})
        self.conv1 = nn.Conv2d(576, 300, 3)
        self.fc1 = nn.Linear(10800, 30)
        self.dr = nn.Dropout()

    def forward(self, x):
        feature_layer = self.feature_extraction(x)['mob_feature']
        x = F.relu(self.conv1(feature_layer))
        x = x.flatten(start_dim=1)
        x = self.dr(x)
        x = self.fc1(x)
        return x

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    
    model = WasteClassificationModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def predict_image(image, model, device, class_names):
    image_transform = transforms.Compose([
        v2.Resize(size=(256, 256)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    image = image.convert('RGB')
    image = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        with autocast():
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1)[0]  # Get probabilities

    group_probs = {group: 0.0 for group in class_groups}

    for group, items in class_groups.items():
        group_probs[group] = probabilities[[class_names.index(item) for item in items]].mean().item()

    sorted_groups = sorted(group_probs.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_groups[:-2]

st.title("Waste Classification")

with st.expander("About"):
    st.write("""Introducing our AI-powered **Waste Classification System**, designed to automate waste sorting operations. By utilizing AI, our system achieves an impressive accuracy rate of **85%**, outperforming traditional manual and semi-mechanical sorting methods.

Our solution enhances operational efficiency by minimizing fatigue-related errors, while significantly lowering energy consumption and costs. Ideal for large-scale waste processing, the AI-driven system optimizes productivity and ensures sustainable waste management, offering a scalable approach to meet the demands of smart cities.""")

st.logo(image="./img/logo.png", size="large")

img_path = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

st.markdown("##### Example Images")

col_1, col_2, col_3 = st.columns(3)

with col_1:
    st.image(example_images["Plastic"], use_container_width=True)
    if st.button("Predict Plastic Image", key="plastic"):
        img_path = example_images["Plastic"]
        
with col_2:
    st.image(example_images["Metal"], use_container_width=True)
    if st.button("Predict Metal Image", key="metal"):
        img_path = example_images["Metal"]
with col_3:
    st.image(example_images["Glass"], use_container_width=True)
    if st.button("Predict Glass Image", key="glass"):
        img_path = example_images["Glass"]

if img_path is not None:
    image = Image.open(img_path).convert('RGB')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    with st.spinner('Loading model...'):
        model = load_model(model_path, device)

    with st.spinner('Making prediction...'):
        predictions = predict_image(image, model, device, class_names)

    
    col1, col2 = st.columns([0.6, 0.4], vertical_alignment="center")
    
    with col1:
        st.image(image, use_container_width=True)

    with col2:
        for i, (label, prob) in enumerate(predictions):
            st.write(f"{i + 1}. {label} ({prob * 100:.2f}%)")
            st.progress(prob)

st.divider()
art = """‎ 
      ________________    ___/-\___     ___/-\___     ___/-\___
     / /             ||  |---------|   |---------|   |---------|
    / /              ||   |       |     | | | | |     |   |   |
   / /             __||   |       |     | | | | |     | | | | |
  / /   \\         I  ||   |       |     | | | | |     | | | | |
 (-------------------||   | | | | |     | | | | |     | | | | |
 ||               == ||   |_______|     |_______|     |_______|
 ||   Kelompok 4 ❤   | ==============================================
 ||          ____    |                                ____      |
( | o      / ____ \\                                 / ____ \\    |)
 ||      / / . . \\ \\                              / / . . \\ \\   |
[ |_____| | .   . | |____________________________| | .   . | |__]
          | .   . |                                | .   . |
           \\_____/                                  \\_____/       """
st.text(art)