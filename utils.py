import os
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math


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

onlymaladies = maladies.copy()
onlymaladies.remove(nofinding)

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


def trace_effectif_maladie(df, gender=False):
    """ Affiche desinfo sur les maladies, separer selon le genre avec gender =True"""
    df = df.copy()
    df = df[[patientgender] + maladies]
    fig = make_subplots(
        rows=3, cols=1, 
        row_heights=[0.65, 0.1, 0.35],  
        subplot_titles=["Maladies détectées", "No Finding", "Répartition des maladies"]
    )
    fig.update_layout(height=700, width=800)

    if gender:
        workingdf = df.groupby(patientgender)[maladies].sum().T.reset_index()
        workingdf.columns = ["Maladie", "F", "M"]
        workingdf = workingdf.melt(id_vars="Maladie", var_name="Sexe", value_name="Effectif")
        
        # pourcentages par genre par maladie
        df_percent = workingdf.copy()
        df_percent["Pourcentage"] = df_percent.groupby("Maladie")["Effectif"].transform(lambda x: 100 * x / x.sum())
        df_percent["Max_Pourcentage"] = df_percent.groupby("Maladie")["Pourcentage"].transform("max")
        df_percent = df_percent.sort_values("Max_Pourcentage", ascending=True)
        df_percent = df_percent.drop(columns=["Max_Pourcentage"])
        fig_pourcent = px.bar( df_percent,  "Maladie",  "Pourcentage",  "Sexe", 
            title="Répartition des patients par genre et maladie (en %) ", 
            color_discrete_map=sexmap,
            labels={"Pourcentage": "Pourcentage", "Maladie": "Maladie"},
            barmode="relative"
        )
        fig_pourcent.add_hline(y=50,line=dict(color="red",dash="dash"))
        fig_pourcent.update_layout(barmode="relative")
    else:
        workingdf = df[maladies].sum().reset_index()
        workingdf.columns = ["Maladie", "Effectif"]

        df_percent = workingdf.copy()
        df_percent["Pourcentage"] = 100 * df_percent["Effectif"] / df_percent["Effectif"].sum()
        df_percent = df_percent[df_percent["Maladie"] != "No Finding"]
        df_percent = df_percent.sort_values("Pourcentage", ascending=True)
        fig_pourcent = px.bar( df_percent,  "Maladie",  "Pourcentage", 
            title="Répartition des patients par genre et maladie (en %) ", 
            color_discrete_map=sexmap,
            labels={"Pourcentage": "Pourcentage", "Maladie": "Maladie"},
            barmode="stack",
        )    
    workingdf_main = workingdf[workingdf["Maladie"] != "No Finding"]
    workingdf_no_finding = workingdf[workingdf["Maladie"] == "No Finding"]


    def barplot(df) :
        return px.bar(
        df, x="Effectif", y="Maladie",
        color="Sexe" if gender else None,
        orientation="h", barmode="group",
        color_discrete_map=sexmap if gender else None,
    )
    
    fig_main = barplot(workingdf_main)
    fig_no_finding = barplot(workingdf_no_finding)

    for trace in fig_main.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in fig_no_finding.data:
        fig.add_trace(trace, row=2, col=1)
    for trace in fig_pourcent.data:
        fig.add_trace(trace, row=3, col=1)

    fig.update_yaxes(categoryorder="total ascending", row=1, col=1)
    fig.update_yaxes(categoryorder="total ascending", row=2, col=1)

    fig.update_layout(height=800, width=800,  
                       title_text="Évolution des maladies selon l'âge",
                      template="plotly_white")
    fig.update_layout(showlegend=True) 
    
    fig.show()

def plot_disease_trends(df, maladies, window=5, gender=False):
    df = df.copy()
    if isinstance(maladies, str):
        maladies = [maladies]

    fig = make_subplots(rows=len(maladies), cols=1,
                        subplot_titles=[f"Évolution de {maladie}" for maladie in maladies],
                        )

    for i, maladie in enumerate(maladies, start=1):
        if gender:
            for sexe in ["M", "F"]:
                df_gender = df[df["Patient Gender"] == sexe]
                df_maladie = df_gender.groupby("Patient Age")[maladie].mean() * 100
                df_maladie = df_maladie.reset_index()
                df_maladie[maladie + '_smooth'] = df_maladie[maladie].rolling(window=window, min_periods=1).mean()

                fig.add_trace(
                    go.Scatter(x=df_maladie["Patient Age"], y=df_maladie[maladie + "_smooth"],
                               mode='lines', line=dict(color=sexmap[sexe]), showlegend=False),
                    row=i, col=1
                )
            fig.add_annotation(
            text="<b>Code couleur :</b> <span style='color:blue'>M = Bleu</span> | <span style='color:red'>F = Rouge</span>",
            x=0.5, y=1.02, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14), align="center")
        else:
            df_maladie = df.groupby("Patient Age")[maladie].mean() * 100
            df_maladie = df_maladie.reset_index()
            df_maladie[maladie + '_smooth'] = df_maladie[maladie].rolling(window=window, min_periods=1).mean()

            fig.add_trace(
                go.Scatter(x=df_maladie["Patient Age"], y=df_maladie[maladie + "_smooth"],
                           mode='lines', showlegend=False),
                row=i, col=1
            )

        fig.update_xaxes(title_text="Âge", row=i, col=1)
        fig.update_yaxes(title_text="% attein", row=i, col=1)    
    
    fig.update_layout(height=150 *(len(maladies)+1), width=1000,  
                       title_text="Évolution des maladies selon l'âge",
                      template="plotly_white", showlegend=False)
    
    fig.show()


def stacked_area_chart(df, maladies, window=5, vlines=None, gender=False):
    vlines = [] if vlines is None else vlines
    df = df.copy()
    if gender:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Évolution des maladies chez les hommes", "Évolution des maladies chez les femmes"],
            shared_yaxes=True
        )
        for i, sexe in enumerate(["M", "F"], start=1):
            df_filtered = df[df[patientgender] == sexe]
            df_age = df_filtered[[patientage] + maladies]
            df_agg = df_age.groupby(patientage).mean() * 100
            df_agg = df_agg.rolling(window=window, min_periods=1).mean().reset_index()
            for maladie in maladies:
                fig.add_trace(
                    go.Scatter(
                        x=df_agg[patientage], y=df_agg[maladie], 
                        fill='tonexty', mode='none', 
                        stackgroup=f"group_{sexe}",
                        name=f"{maladie} ({sexe})"
                    ),
                    row=1, col=i
                )
            for v in vlines:
                fig.add_vline(v, line_dash="dash", line_color="red", row=1, col=i)

        fig.update_layout(
            title_text="Évolution des maladies selon l'âge et le sexe",
            template="plotly_white",
            showlegend=True
        )

    else:
        df_age = df[[patientage] + maladies]  # Colonnes d'intérêt
        df_agg = df_age.groupby(patientage).mean() * 100  # Moyenne des maladies par âge
        df_agg = df_agg.rolling(window=window, min_periods=1).mean().reset_index()

        fig = px.area(df_agg, x=patientage, y=maladies,
                      labels={"value": "% de patients", patientage: "Âge"},
                      title="Évolution des maladies selon l'âge",
                      template="plotly_white")

        for v in vlines:
            fig.add_vline(v, line_dash="dash", line_color="red")

    fig.show()



def plot_avg_diseases_by_age(df, maladies, window_size=5, vlines=None, gender=False):
    """Moyenne du nombre de maladies par âge, avec option de séparer par sexe et ajout de couleurs"""
    vlines = [] if vlines is None else vlines
    df = df.copy()
    nbmaladie = "Nombre de maladies"
    df[nbmaladie] = df[maladies].sum(axis=1)

    if gender:
        df_fem = df[df[patientgender] == F]
        df_agg_fem = df_fem.groupby(patientage)[nbmaladie].mean().reset_index()
        df_agg_fem[nbmaladie] = df_agg_fem[nbmaladie].rolling(window=window_size, min_periods=1).mean()

        df_masc = df[df[patientgender] == M]
        df_agg_masc = df_masc.groupby(patientage)[nbmaladie].mean().reset_index()
        df_agg_masc[nbmaladie] = df_agg_masc[nbmaladie].rolling(window=window_size, min_periods=1).mean()

        fig = px.line(
            df_agg_fem, x=patientage, y=nbmaladie, 
            labels={nbmaladie: "Moyenne du nombre de maladies", patientage: "Âge"},
            title="Nombre moyen de maladies par âge",
            template="plotly_white",
            color_discrete_sequence=[sexmap[F]]
        )

        fig.add_traces(px.line(
            df_agg_masc, x=patientage, y=nbmaladie, 
            color_discrete_sequence=[sexmap[M]]
        ).data)

        fig.add_annotation(
            text="<b>Code couleur :</b> <span style='color:blue'>M = Bleu</span> | <span style='color:red'>F = Rouge</span>",
            x=0.5, y=1.02, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14), align="center")

    else:
        # Si pas de séparation par sexe, on fait un seul groupe
        df_agg_maladie = df.groupby(patientage)[nbmaladie].mean().reset_index()
        df_agg_maladie[nbmaladie] = df_agg_maladie[nbmaladie].rolling(window=window_size, min_periods=1).mean()
        fig = px.line(
            df_agg_maladie, x=patientage, y=nbmaladie,
            labels={nbmaladie: "Moyenne du nombre de maladies", patientage: "Age"},
            title="Nombre moyen de maladies par âge",
            template="plotly_white"
        )

    for v in vlines:
        fig.add_vline(v, line_dash="dash", line_color="red")

    fig.show()


def plot_patient_age_distribution(df, max_diseases_per_plot=18):
    d = df.copy()
    d= d.drop(columns=[x for x in d.columns if x not in maladies and x!="Patient Age"])
    df_melted = d.melt(id_vars=["Patient Age"], var_name="Disease", value_name="Has Disease")
    df_melted = df_melted[df_melted["Has Disease"] == 1]
    
    unique_diseases = df_melted["Disease"].unique()
    num_plots = math.ceil(len(unique_diseases) / max_diseases_per_plot)

    for i in range(num_plots):
        diseases_subset = unique_diseases[i * max_diseases_per_plot: (i + 1) * max_diseases_per_plot]
        df_subset = df_melted[df_melted["Disease"].isin(diseases_subset)]

       
        fig = px.box(df_subset, x="Disease", y="Patient Age", 
                     labels={"Patient Age": "Âge du patient", "Disease": "Maladie"},
                     title=f"Distribution de l'âge des patients par maladie",
                     color="Disease")  
        fig.update_layout(xaxis_tickangle=-45)

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


