RNG_SEED = 2025
MAX_AGE = 120
sexmap = {"M": "blue", "F": "red"}
M, F = "M", "F"
patientid = "Patient ID"
patientage = "Patient Age"
patientgender = "Patient Gender"
# 'OriginalImage[Width','Height]', 'OriginalImagePixelSpacing[x', 'y]',
viewposition = "View Position"

findinglabels="Finding Labels"

maladies = [
    atelectasis := "Atelectasis",
    cardiomegaly := "Cardiomegaly",
    consolidation := "Consolidation",
    edema := "Edema",
    effusion := "Effusion",
    emphysema := "Emphysema",
    fibrosis := "Fibrosis",
    hernia := "Hernia",
    infiltration := "Infiltration",
    mass := "Mass",
    nofinding := "No Finding",
    nodule := "Nodule",
    pleural_thickening := "Pleural_Thickening",
    pneumonia := "Pneumonia",
    pneumothorax := "Pneumothorax",
]

onlymaladies = maladies.copy()
onlymaladies.remove(nofinding)

