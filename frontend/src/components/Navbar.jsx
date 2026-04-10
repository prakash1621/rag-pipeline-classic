import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

function Navbar() {
  const { user, isAdmin, logout } = useAuth();

  return (
    <nav aria-label="Main navigation">
      <ul>
        <li>
          <Link to="/query">Query</Link>
        </li>
        {isAdmin && (
          <li>
            <Link to="/admin">Admin</Link>
          </li>
        )}
      </ul>
      <div className="nav-user">
        <span>{user?.username}</span>
        <button onClick={logout}>Logout</button>
      </div>
    </nav>
  );
}

export default Navbar;
