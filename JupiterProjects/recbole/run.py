# para ejecutar el modelo 
from recbole.quick_start import run_recbole
#run_recbole(dataset='bgg', model='ItemKNN', config_file_list=['bgg.yaml'])
dataset = str(input('Los datos(bgg): '))
model = str(input('tipo de modelo: '))
run_recbole(dataset=dataset, model=model, config_file_list=['bgg.yaml'])
#run_recbole(dataset='bgg', model='Random', config_file_list=['bgg.yaml'])