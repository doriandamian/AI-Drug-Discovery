# Etapa 1: Construirea aplicației Angular
FROM node:22-alpine AS build
WORKDIR /app

# Copiem fișierele de configurare și instalăm dependențele
COPY package*.json ./
RUN npm install

# Copiem restul codului și compilăm aplicația pentru producție
COPY . .
RUN npm run build -- --configuration production

# Etapa 2: Servirea aplicației cu Nginx
FROM nginx:stable-alpine

# Copiem fișierele compilate din etapa anterioară în folderul Nginx
# Notă: Verifică în folderul 'dist/' numele exact al proiectului tău
COPY --from=build /app/dist/frontend/browser /usr/share/nginx/html

# Copiem o configurație personalizată de Nginx pentru a suporta rutele de Angular
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]