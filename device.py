import torch
print(torch.cuda.is_available())  # Deve stampare True
print(torch.cuda.device_count())  # Deve essere almeno 1
print(torch.cuda.get_device_name(0))  # Dovrebbe stampare il nome della scheda grafica dedicata
