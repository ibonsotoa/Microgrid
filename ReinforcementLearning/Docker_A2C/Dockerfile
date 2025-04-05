FROM python:3.9-slim

WORKDIR /app

# Copia los archivos del proyecto
COPY . .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto 5000
EXPOSE 5000

# Comando de inicio: ejecuta el servidor
CMD ["python", "server.py"]
