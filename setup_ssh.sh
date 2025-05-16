git clone https://github.com/namanhboi/ParlayANN.git

cd ParlayANN

git fetch

git checkout dna/distance_calc

git submodule init
git submodule upate


sudo apt-get update
sudo apt-get install python3.10-venv

python3.10 -m venv env

source env/bin/activate

pip install optuna
pip install matplotlib

cd algorithms/vamana
make
