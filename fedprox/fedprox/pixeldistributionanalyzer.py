import numpy as np
import torch
from typing import Tuple
# Nous assumons que matplotlib est disponible pour le plotting
# Si ce code est exécuté dans un environnement sans affichage, le plotting peut échouer.
try:
    import matplotlib.pyplot as plt
    # S'assurer que le backend est non interactif pour la sauvegarde de fichiers
    plt.switch_backend('Agg') 
except ImportError:
    print("Avertissement: Matplotlib n'est pas installé. La visualisation sera désactivée.")
    plt = None


# Ces fonctions sont conçues pour être des méthodes d'une classe Client 
# qui possède self.client_id et self.traindata (un DataLoader)
# Pour l'exemple, nous les mettons dans une classe utilitaire.
class PixelDistributionAnalyzer:
    """Contient les méthodes pour calculer et visualiser la distribution des pixels 
    pour un client donné, en tenant compte de la standardisation des données.
    """
    
    def __init__(self, client_id, traindata_loader):
        self.client_id = client_id
        self.traindata = traindata_loader
        
    def _calculate_pixel_distribution(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Calcule l'histogramme des pixels sur la plage [-5.0, 5.0], 
        adaptée aux données standardisées et décalées.
        """
        all_pixels = []
        print(f"Client {self.client_id}: Démarrage du calcul de la distribution des pixels standardisés...")

        # Itérer sur le DataLoader
        try:
            for images, _ in self.traindata:
                # Les images sont déjà des tenseurs normalisés/standardisés et décalés.
                if isinstance(images, torch.Tensor):
                    # Convertir en NumPy. Les valeurs sont autour de 0, pas 0-255.
                    images_np = images.cpu().numpy()
                elif isinstance(images, np.ndarray):
                    images_np = images
                else:
                    continue
                
                # NOUVEAU: PAS DE MULTIPLICATION PAR 255. Les valeurs sont utilisées telles quelles.
                
                # Aplatir et ajouter tous les pixels
                all_pixels.append(images_np.flatten())
                
        except Exception as e:
            print(f"Erreur lors de l'itération sur le DataLoader du client {self.client_id}: {e}")
            return np.array([]), np.array([]), 0.0, 0.0

        if not all_pixels:
            print(f"Client {self.client_id}: Aucun pixel trouvé.")
            return np.array([]), np.array([]), 0.0, 0.0
            
        all_pixels_flat = np.concatenate(all_pixels).astype(np.float32)
        
        # MODIFICATION CRITIQUE: Calculer l'histogramme sur la nouvelle plage [-5.0, 5.0]
        HISTOGRAM_RANGE = (-5.0, 5.0)
        NUM_BINS = 100 # Plus de bins pour une meilleure visualisation du décalage
        
        hist, bin_edges = np.histogram(all_pixels_flat, bins=NUM_BINS, range=HISTOGRAM_RANGE)
        
        # Calculer les statistiques clés
        mean_intensity = np.mean(all_pixels_flat) if all_pixels_flat.size > 0 else 0.0
        std_intensity = np.std(all_pixels_flat) if all_pixels_flat.size > 0 else 0.0

        print(f"Client {self.client_id}: Calcul terminé. Moyenne standardisée: {mean_intensity:.4f}, Écart-type: {std_intensity:.4f}")

        return hist, bin_edges, mean_intensity, std_intensity

    def visualize_pixel_distribution(self, round_number: int):
        """
        Visualise la distribution des pixels standardisés, imprime les statistiques et sauve le graphique.
        """
        hist, bin_edges, mean, std = self._calculate_pixel_distribution()
        
        if hist.size == 0:
            print(f"Client {self.client_id}: Impossible de visualiser. Distribution non calculée.")
            return

        print("\n" + "="*80)
        print(f"--- Client {self.client_id} - Distribution des Pixels Standardisés (Round {round_number}) ---")
        # Le décalage de domaine se manifeste par une variation de ces deux valeurs
        print(f"Moyenne d'intensité standardisée: {mean:.4f}")
        print(f"Écart-type d'intensité standardisée: {std:.4f}")
        print("Les variations de moyenne et d'écart-type confirment le décalage de domaine.")
        print("="*80 + "\n")

        # --- Partie de Plotting ---
        if plt is not None:
            try:
                plt.figure(figsize=(8, 5))
                # Utiliser plt.bar pour afficher l'histogramme calculé
                plt.bar(
                    bin_edges[:-1], 
                    hist, 
                    width=(bin_edges[1] - bin_edges[0]), # Largeur ajustée à la bin
                    color='#3F51B5', # Couleur bleue pour mieux contraster
                    edgecolor='black', 
                    alpha=0.7
                )
                
                plt.axvline(mean, color='#FF5722', linestyle='dashed', linewidth=2, label=f'Moyenne: {mean:.4f}')
                
                # Mise à jour des titres et étiquettes
                plt.title(f'Distribution des Pixels Standardisés - Client {self.client_id} (Round {round_number})', fontsize=14)
                plt.xlabel('Intensité des Pixels Standardisés (Plage typique [-3.0, 3.0])', fontsize=12)
                plt.ylabel('Fréquence (Nombre de Pixels)', fontsize=12)
                plt.legend()
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                
                # Sauvegarde de la figure
                plot_filename = f"client_{self.client_id}_pixel_dist_r{round_number}.png"
                plt.savefig(plot_filename)
                plt.close() 
                print(f"Client {self.client_id}: Graphique de distribution sauvegardé sous {plot_filename}")

            except Exception as e:
                print(f"Client {self.client_id}: Avertissement - Erreur lors de la sauvegarde du graphique : {e}")
        else:
            print("Client {self.client_id}: Le plotting est désactivé (Matplotlib manquant ou erreur).")
