import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Custom Dataset
class Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepaths']
        label = int(self.dataframe.iloc[idx]['labels'])  
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            
        return image, label, img_path

def split_dataset(benign_dir, malignant_dir):
    # Create training and test datasets
    filepaths = []
    labels = []
    for label_dir, label in [(benign_dir, '0'), (malignant_dir, '1')]:
        files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        filepaths.extend(files)
        labels.extend([label] * len(files))
    df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})
    train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['labels'])

    print("Train dataset:", train_df.shape[0])
    print("Test dataset:", test_df.shape[0])

    train_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomRotation(40),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = Dataset(train_df, transform=train_transform)
    test_dataset = Dataset(test_df, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    return train_loader, test_loader
