import os



RNG_SEED = 2025
MAX_AGE = 120
sexmap = {"M": "blue", "F": "red"}
M, F = "M", "F"
patientid = "Patient ID"
patientage = "Patient Age"
patientgender = "Patient Gender"
# 'OriginalImage[Width','Height]', 'OriginalImagePixelSpacing[x', 'y]',
viewposition = "View Position"

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


# Charger les variables du fichier .env
def load_env_file(env_path=".env"):
    if os.path.exists(env_path):
        with open(env_path, "r") as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#"):  # Ignorer les commentaires et lignes vides
                    key, value = line.split("=", 1)
                    os.environ[key] = value  # Définir la variable d'environnement
                    
def plot_age_dist(df, title="Distribution des ages", gender=False, vlines=None):
    from plotly.subplots import make_subplots # type: ignore
    import plotly.graph_objects as go # type: ignore
    """ Affiche la distibution des ages, separer selon le genre avec gender=True"""
    fig = go.Figure()
    df = df.copy()
    vlines = [] if vlines is None else vlines
    
    if not gender:  # distribution unique
        df_grouped = df.groupby(patientage).size().reset_index(name="Count")
        fig.add_trace(go.Bar(x=df_grouped[patientage], y=df_grouped["Count"]))
        for v in vlines:
            fig.add_vline(v, line_dash="dash", line_color="red")
    else:  # distribution par genre
        df_grouped = df.groupby([patientage, patientgender]).size().reset_index(name="Count")
        fig = make_subplots(
            1, 2, subplot_titles=("Masculins", "Féminins"), shared_yaxes=True
        )
        df_men = df_grouped[df_grouped[patientgender] == "M"]
        df_women = df_grouped[df_grouped[patientgender] == "F"]
        
        fig.add_trace(go.Bar(x=df_men[patientage], y=df_men["Count"]), 1, 1)
        fig.add_trace(go.Bar(x=df_women[patientage], y=df_women["Count"]), 1, 2)
        for v in vlines:
            fig.add_vline(v, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title=title,
        xaxis_title="Age",
        yaxis_title="Effectif",
        showlegend=False,
    )
    fig.show()    



if __name__ == "__main__":
    load_env_file()
    data_path = os.getenv("DATA_DIR", "data/default/")
    print(f"Les données sont dans : {data_path}")
    data_dir = os.getenv("DATA_DIR", "data/default/")
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print(f"Sous-dossiers de {data_dir} : {subdirs}")
    else:
        print(f"Le dossier {data_dir} n'existe pas ou n'est pas un dossier.")


