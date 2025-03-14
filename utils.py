import os

# Charger les variables du fichier .env
def load_env_file(env_path=".env"):
    if os.path.exists(env_path):
        with open(env_path, "r") as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#"):  # Ignorer les commentaires et lignes vides
                    key, value = line.split("=", 1)
                    os.environ[key] = value  # Définir la variable d'environnement

# Charger le fichier .env
load_env_file()


if __name__ == "__main__":
    data_path = os.getenv("DATA_DIR", "data/default/")
    print(f"Les données sont dans : {data_path}")
    data_dir = os.getenv("DATA_DIR", "data/default/")
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print(f"Sous-dossiers de {data_dir} : {subdirs}")
    else:
        print(f"Le dossier {data_dir} n'existe pas ou n'est pas un dossier.")


