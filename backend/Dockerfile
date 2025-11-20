# backend/Dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --production
COPY . .
ENV NODE_ENV=production
EXPOSE 4000
CMD ["node","server.js"]
