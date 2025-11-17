# Importar los módulos necesarios
import redis
import datetime

# Crear una conexión a redis
r = redis.Redis(host='localhost', port=6379)

# Obtener las 10 últimas muestras de la serie temporal temperatura
# El - indica comenzar desde el tiempo más temprano (antiguo) al + que indica el tiempo más reciente
muestras = r.execute_command('TS.REVRANGE', 'temperature', '-', '+', 'COUNT', 10)

# Recorrer la lista de muestras
for m in muestras:
  # Extraer el valor y el timestamp
  valor = m[1]
  timestamp = m[0]

  # Dividir el timestamp por 1000 para obtener los segundos
  timestamp = timestamp / 1000

  # Convertir el timestamp en un objeto datetime
  dt = datetime.datetime.fromtimestamp(timestamp)

  # Formatear el objeto datetime en una cadena
  dt_str = dt.strftime('%d/%m/%Y %H:%M:%S')

  # Imprimir el valor y la fecha y hora
  print(f'Temperatura: {valor} °C - Fecha y hora: {dt_str}')
