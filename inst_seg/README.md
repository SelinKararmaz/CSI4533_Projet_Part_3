
# Instance Segmentation

Ce projet a été testé avec Python 3.10.

## Pour Commencer

Suivez ces étapes simples pour obtenir une copie locale en fonctionnement.

### Prérequis

Ce projet utilise GIT pour le contrôle de version. Assurez-vous que GIT est installé sur votre ordinateur. Pour les instructions d'installation, visitez :

[https://git-scm.com/book/fr/v2/Démarrage-rapide-Installation-de-Git](https://git-scm.com/book/fr/v2/Démarrage-rapide-Installation-de-Git)

### Installation

1. Dans le Terminal ou le CommandLine, clonez le dépôt
   ```
   git clone https://gitlab.com/zatoitche/inst_seg.git
   ```

2. Téléchargez le fichier `images.zip` pour la deuxième partie de votre laboratoire depuis le lien suivant :
   ```
   https://drive.google.com/file/d/1potC4tmKjvLAlXSmhaGH59u-g5u4qg5-/view?usp=sharing
   ```

3. Extrayez `images.zip` et placez le dossier "images" extrait dans le répertoire du projet que vous avez précédemment cloné.

4. Dans le Terminal ou le CommandLine, naviguez vers votre répertoire de projet.

5. Créez un environnement virtuel nommé "env" en utilisant la commande suivante :
   ```
   python3 -m venv env
   ```

6. Activez l'environnement virtuel avec :
   ```
   source env/bin/activate
   ```

7. Installez les dépendances du code avec :
   ```
   pip install -r requirements.txt
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

## Utilisation

Vous êtes maintenant prêt à exécuter le code. Le fichier `main.py` contient un exemple de code pour effectuer la segmentation d'instance sur des personnes dans une image. Le code cible les images dans le dossier "examples", traitant les images de "examples/source" et les plaçant dans "examples/output" pour vos tests. Vous pouvez modifier le code pour traiter les images du dossier "images" pour votre projet.
