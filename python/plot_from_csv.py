import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_logarithmic(csv_file, col1, col2, save_plot=None):
    # Lire le fichier CSV
    df = pd.read_csv(csv_file)
    
    # Vérifier si les colonnes spécifiées existent dans le fichier CSV
    if col1 not in df.columns or col2 not in df.columns:
        print(f"Les colonnes spécifiées '{col1}' ou '{col2}' ne sont pas présentes dans le fichier CSV.")
        return
    
    # Extraire les données des deux colonnes
    x = df[col1]
    y = df[col2]
    
    # Vérifier s'il y a des valeurs <= 0 pour une échelle logarithmique
    if (y <= 0).any():
        print("Les données contiennent des valeurs non positives dans la colonne y, ce qui n'est pas permis pour une échelle logarithmique.")
        return
    
    # Tracer les données avec une échelle logarithmique sur l'axe vertical
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.yscale('log')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(f'Plot logarithmique de {col1} vs {col2}')
    plt.grid(True, which="both", ls="--")
    
    # Régler l'axe X pour qu'il soit par pas de 32
    plt.xticks(np.arange(0, max(x) + 1, 64))
    # Enregistrer le plot si un chemin est spécifié
    if save_plot:
        plt.savefig(save_plot)
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trace un plot logarithmique à partir d'un fichier CSV et deux colonnes spécifiées.")
    parser.add_argument("csv_file", type=str, help="Chemin vers le fichier CSV")
    parser.add_argument("col1", type=str, help="Nom de la première colonne")
    parser.add_argument("col2", type=str, help="Nom de la deuxième colonne")
    parser.add_argument("--save_plot", type=str, help="Chemin où enregistrer le plot (optionnel)")
    args = parser.parse_args()

    plot_logarithmic(args.csv_file, args.col1, args.col2, args.save_plot)
