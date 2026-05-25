FROM node:18-alpine

WORKDIR /app

# Copiem doar fișierele de dependințe pentru a folosi cache-ul Docker
COPY package*.json ./
RUN npm install

# Copiem restul codului
COPY . .

# Expunem portul standard de Angular
EXPOSE 4200

# Comanda default (va fi oricum suprascrisă de cea din docker-compose.yml)
CMD ["npx", "ng", "serve", "--host", "0.0.0.0"]