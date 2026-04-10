import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api';

const AuthContext = createContext(null);

/**
 * Decode the payload segment of a JWT (base64url-encoded middle part).
 * Returns the parsed JSON object or null on failure.
 */
function decodeJwtPayload(token) {
  try {
    const base64Url = token.split('.')[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split('')
        .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    );
    return JSON.parse(jsonPayload);
  } catch {
    return null;
  }
}

/**
 * Check whether a decoded JWT payload is expired.
 */
function isTokenExpired(payload) {
  if (!payload || !payload.exp) return true;
  return Date.now() >= payload.exp * 1000;
}

function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [accessToken, setAccessToken] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  const logout = useCallback(() => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    setUser(null);
    setAccessToken(null);
    navigate('/login');
  }, [navigate]);

  const login = useCallback(async (username, password) => {
    const { data } = await api.post('/api/auth/login', { username, password });
    const { access_token, refresh_token } = data;
    localStorage.setItem('access_token', access_token);
    localStorage.setItem('refresh_token', refresh_token);
    setAccessToken(access_token);

    const payload = decodeJwtPayload(access_token);
    setUser({ username: payload.sub, role: payload.role });
  }, []);

  // On mount: restore session from localStorage tokens
  useEffect(() => {
    const restoreSession = async () => {
      const storedToken = localStorage.getItem('access_token');
      if (!storedToken) {
        setLoading(false);
        return;
      }

      const payload = decodeJwtPayload(storedToken);

      if (payload && !isTokenExpired(payload)) {
        setAccessToken(storedToken);
        setUser({ username: payload.sub, role: payload.role });
        setLoading(false);
        return;
      }

      // Token is expired or invalid — attempt refresh
      const refreshToken = localStorage.getItem('refresh_token');
      if (!refreshToken) {
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        setLoading(false);
        return;
      }

      try {
        const { data } = await api.post('/api/auth/refresh', {
          refresh_token: refreshToken,
        });
        const newToken = data.access_token;
        localStorage.setItem('access_token', newToken);
        setAccessToken(newToken);

        const newPayload = decodeJwtPayload(newToken);
        setUser({ username: newPayload.sub, role: newPayload.role });
      } catch {
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
      } finally {
        setLoading(false);
      }
    };

    restoreSession();
  }, []);

  const isAdmin = user?.role === 'admin';

  return (
    <AuthContext.Provider
      value={{ user, accessToken, login, logout, isAdmin, loading }}
    >
      {children}
    </AuthContext.Provider>
  );
}

const useAuth = () => useContext(AuthContext);

export { AuthContext, AuthProvider, useAuth };
