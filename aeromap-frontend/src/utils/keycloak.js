import Keycloak from 'keycloak-js';

const keycloak = new Keycloak({
  url: 'http://localhost:8080',
  realm: 'aviation-realm',
  clientId: 'aviation-api'
});

export const initKeycloak = async () => {
  try {
    const auth = await keycloak.init({
      onLoad: true,
      pkceMethod: 'S256',
      checkLoginIframe: false,  // Фикс: Отключаем iframe для избежания CSP/timeout
      messageReceiveTimeout: 30000  // Опционально: Увеличьте timeout на 30 сек
    });
    if (auth) {
      localStorage.setItem('access_token', keycloak.token);
    }
    return auth;
  } catch (error) {
    console.error('Keycloak init error:', error);
    return false;
  }
};

export default keycloak;
