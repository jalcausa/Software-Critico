# 1. Usar una imagen base de Python
FROM python:3.11-slim

# 2. Establecer el directorio de trabajo
WORKDIR /app

# 3. Copiar el archivo de requisitos
COPY requirements.txt .

# 4. Instalar las dependencias
RUN pip install -r requirements.txt

# 5. Copiar los archivos del modelo y configuración
COPY modelo.keras .
COPY config.json .

# 6. Copiar el resto del código de la aplicación
COPY . .

# 7. Establecer el puerto en el que correrá la app DENTRO del contenedor
#  app.py usa os.getenv('PORT', ...), así que leerá esta variable.
ENV PORT=80

# 8. Exponer el puerto 80 para que Docker sepa que el contenedor escucha ahí
EXPOSE 80

# 9. Comando para ejecutar la aplicación
CMD ["python", "app.py"]


