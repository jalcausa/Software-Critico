# Importar los módulos necesarios
import redis
import random
import time

# Crear una conexión a redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Crear una serie temporal llamada 'temperature'
try:
    r.execute_command('TS.CREATE', 'temperature')
except Exception as e:
   print(e)
   

# Generar un bucle infinito
while True:
  # Generar un valor aleatorio de temperatura entre 10 y 40 grados
  temp = random.randint(10, 40)

  # Añadir el valor a la serie temporal con la hora actual (el * indica añadir marca automáticamente)
  r.execute_command('TS.ADD', 'temperature', '*', temp)

  # Imprimir el valor generado
  print(f'Temperature: {temp} °C')

  # Esperar 10 segundos
  time.sleep(10)
