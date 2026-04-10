import { useState, useEffect, useCallback } from 'react';
import api from '../api';

function AdminPage() {
  const [notification, setNotification] = useState(null);

  // Knowledge Base state
  const [kbLoading, setKbLoading] = useState(false);

  // Documents state
  const [documents, setDocuments] = useState({});
  const [docsLoading, setDocsLoading] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadCategory, setUploadCategory] = useState('');

  // Users state
  const [users, setUsers] = useState([]);
  const [usersLoading, setUsersLoading] = useState(false);
  const [newUsername, setNewUsername] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [newRole, setNewRole] = useState('viewer');

  // Provider config state
  const [providerConfig, setProviderConfig] = useState(null);

  const showNotification = (message, type = 'success') => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 5000);
  };

  // ── Fetch helpers ──────────────────────────────────────────

  const fetchDocuments = useCallback(async () => {
    setDocsLoading(true);
    try {
      const { data } = await api.get('/api/documents');
      setDocuments(data.documents || {});
    } catch (err) {
      showNotification(err.response?.data?.detail || 'Failed to load documents', 'error');
    } finally {
      setDocsLoading(false);
    }
  }, []);

  const fetchUsers = useCallback(async () => {
    setUsersLoading(true);
    try {
      const { data } = await api.get('/api/users');
      setUsers(Array.isArray(data) ? data : []);
    } catch (err) {
      showNotification(err.response?.data?.detail || 'Failed to load users', 'error');
    } finally {
      setUsersLoading(false);
    }
  }, []);

  const fetchConfig = useCallback(async () => {
    try {
      const { data } = await api.get('/api/config');
      setProviderConfig(data);
    } catch (err) {
      showNotification(err.response?.data?.detail || 'Failed to load config', 'error');
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
    fetchUsers();
    fetchConfig();
  }, [fetchDocuments, fetchUsers, fetchConfig]);

  // ── Knowledge Base handlers ────────────────────────────────

  const handleRebuild = async () => {
    setKbLoading(true);
    try {
      const { data } = await api.post('/api/knowledge-base/rebuild');
      showNotification(
        `Knowledge base rebuilt: ${data.chunks} chunks across ${data.categories} categories`
      );
    } catch (err) {
      showNotification(err.response?.data?.detail || 'Rebuild failed', 'error');
    } finally {
      setKbLoading(false);
    }
  };

  const handleClear = async () => {
    setKbLoading(true);
    try {
      await api.post('/api/knowledge-base/clear');
      showNotification('Knowledge base cleared');
    } catch (err) {
      showNotification(err.response?.data?.detail || 'Clear failed', 'error');
    } finally {
      setKbLoading(false);
    }
  };

  // ── Document handlers ──────────────────────────────────────

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!uploadFile || !uploadCategory.trim()) return;

    const formData = new FormData();
    formData.append('file', uploadFile);
    formData.append('category', uploadCategory.trim());

    try {
      await api.post('/api/documents/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      showNotification('Document uploaded');
      setUploadFile(null);
      setUploadCategory('');
      fetchDocuments();
    } catch (err) {
      showNotification(err.response?.data?.detail || 'Upload failed', 'error');
    }
  };

  const handleDeleteDocument = async (category, filename) => {
    try {
      await api.delete(`/api/documents/${encodeURIComponent(category)}/${encodeURIComponent(filename)}`);
      showNotification('Document deleted');
      fetchDocuments();
    } catch (err) {
      showNotification(err.response?.data?.detail || 'Delete failed', 'error');
    }
  };

  // ── User handlers ──────────────────────────────────────────

  const handleCreateUser = async (e) => {
    e.preventDefault();
    if (!newUsername.trim() || !newPassword) return;

    try {
      await api.post('/api/users', {
        username: newUsername.trim(),
        password: newPassword,
        role: newRole,
      });
      showNotification(`User "${newUsername.trim()}" created`);
      setNewUsername('');
      setNewPassword('');
      setNewRole('viewer');
      fetchUsers();
    } catch (err) {
      showNotification(err.response?.data?.detail || 'Failed to create user', 'error');
    }
  };

  const handleDeleteUser = async (username) => {
    try {
      await api.delete(`/api/users/${encodeURIComponent(username)}`);
      showNotification(`User "${username}" deleted`);
      fetchUsers();
    } catch (err) {
      showNotification(err.response?.data?.detail || 'Failed to delete user', 'error');
    }
  };

  // ── Render ─────────────────────────────────────────────────

  return (
    <div className="admin-page">
      <h1>Admin Dashboard</h1>

      {notification && (
        <div
          className={`notification notification--${notification.type}`}
          role="alert"
        >
          {notification.message}
        </div>
      )}

      {/* Knowledge Base Controls */}
      <section className="admin-section" aria-labelledby="kb-heading">
        <h2 id="kb-heading">Knowledge Base</h2>
        <div className="admin-actions">
          <button onClick={handleRebuild} disabled={kbLoading}>
            {kbLoading ? 'Processing…' : 'Rebuild'}
          </button>
          <button onClick={handleClear} disabled={kbLoading}>
            {kbLoading ? 'Processing…' : 'Clear'}
          </button>
        </div>
      </section>

      {/* Document Manager */}
      <section className="admin-section" aria-labelledby="docs-heading">
        <h2 id="docs-heading">Documents</h2>

        <form onSubmit={handleUpload} className="admin-form" aria-label="Upload document">
          <div className="form-field">
            <label htmlFor="doc-file">File</label>
            <input
              id="doc-file"
              type="file"
              onChange={(e) => setUploadFile(e.target.files[0] || null)}
            />
          </div>
          <div className="form-field">
            <label htmlFor="doc-category">Category</label>
            <input
              id="doc-category"
              type="text"
              value={uploadCategory}
              onChange={(e) => setUploadCategory(e.target.value)}
              placeholder="e.g. HR_Policies"
            />
          </div>
          <button type="submit" disabled={!uploadFile || !uploadCategory.trim()}>
            Upload
          </button>
        </form>

        {docsLoading ? (
          <p>Loading documents…</p>
        ) : Object.keys(documents).length === 0 ? (
          <p>No documents found.</p>
        ) : (
          Object.entries(documents).map(([category, files]) => (
            <div key={category} className="doc-category">
              <h3>{category}</h3>
              <ul>
                {files.map((filename) => (
                  <li key={filename}>
                    <span>{filename}</span>
                    <button
                      onClick={() => handleDeleteDocument(category, filename)}
                      aria-label={`Delete ${filename}`}
                    >
                      Delete
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))
        )}
      </section>

      {/* User Manager */}
      <section className="admin-section" aria-labelledby="users-heading">
        <h2 id="users-heading">Users</h2>

        <form onSubmit={handleCreateUser} className="admin-form" aria-label="Create user">
          <div className="form-field">
            <label htmlFor="new-username">Username</label>
            <input
              id="new-username"
              type="text"
              value={newUsername}
              onChange={(e) => setNewUsername(e.target.value)}
              required
            />
          </div>
          <div className="form-field">
            <label htmlFor="new-password">Password</label>
            <input
              id="new-password"
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              required
            />
          </div>
          <div className="form-field">
            <label htmlFor="new-role">Role</label>
            <select
              id="new-role"
              value={newRole}
              onChange={(e) => setNewRole(e.target.value)}
            >
              <option value="admin">Admin</option>
              <option value="viewer">Viewer</option>
            </select>
          </div>
          <button type="submit" disabled={!newUsername.trim() || !newPassword}>
            Create User
          </button>
        </form>

        {usersLoading ? (
          <p>Loading users…</p>
        ) : users.length === 0 ? (
          <p>No users found.</p>
        ) : (
          <ul className="user-list">
            {users.map((u) => (
              <li key={u.username}>
                <span>
                  {u.username} <em>({u.role})</em>
                </span>
                <button
                  onClick={() => handleDeleteUser(u.username)}
                  aria-label={`Delete user ${u.username}`}
                >
                  Delete
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      {/* Provider Config */}
      <section className="admin-section" aria-labelledby="config-heading">
        <h2 id="config-heading">Provider Configuration</h2>
        {providerConfig ? (
          <dl className="config-list">
            <dt>Embedding Provider</dt>
            <dd>{providerConfig.embedding_provider}</dd>
            <dt>LLM Provider</dt>
            <dd>{providerConfig.llm_provider}</dd>
            <dt>Vector Store Provider</dt>
            <dd>{providerConfig.vector_store_provider}</dd>
          </dl>
        ) : (
          <p>Loading configuration…</p>
        )}
      </section>
    </div>
  );
}

export default AdminPage;
