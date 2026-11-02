import { Layout, Menu } from 'antd'
import { Link, Route, Routes, useLocation } from 'react-router-dom'
import Chat from './pages/Chat'
import Library from './pages/Library'
import PaperDetail from './pages/PaperDetail'

const { Header, Content } = Layout

export default function App() {
  const { pathname } = useLocation()
  const selected = pathname.startsWith('/chat') ? 'chat' : 'library'
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
        <div style={{ color: '#fff', fontWeight: 600, fontSize: 18, whiteSpace: 'nowrap' }}>
          CourseRAG · 论文库
        </div>
        <Menu
          theme="dark"
          mode="horizontal"
          selectedKeys={[selected]}
          style={{ flex: 1, minWidth: 0 }}
          items={[
            { key: 'library', label: <Link to="/">论文库</Link> },
            { key: 'chat', label: <Link to="/chat">问答</Link> },
          ]}
        />
      </Header>
      <Content style={{ padding: 24, maxWidth: 1280, width: '100%', margin: '0 auto' }}>
        <Routes>
          <Route path="/" element={<Library />} />
          <Route path="/paper/:id" element={<PaperDetail />} />
          <Route path="/chat" element={<Chat />} />
        </Routes>
      </Content>
    </Layout>
  )
}
