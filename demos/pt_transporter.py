import socket
import torch
import io

class PTSender:
    def __init__(self, server_ip='127.0.0.1', port=7007):
        self.server_ip = server_ip
        self.port = port

    def send(self, obj):
        """
        将任意 PyTorch 支持的对象（如 list/dict 包含 tensor）发送到云端
        """
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        buffer.seek(0)
        data = buffer.read()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((self.server_ip, self.port))
            client_socket.sendall(data)

        print(f"[PTSender] 已发送 {len(data)} 字节到 {self.server_ip}:{self.port}")


class PTReceiver:
    def __init__(self, listen_ip='0.0.0.0', port=7007):
        self.listen_ip = listen_ip
        self.port = port

    def receive(self):
        """
        接收来自端侧的 PyTorch 序列化对象并反序列化返回
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((self.listen_ip, self.port))
            server_socket.listen(1)
            print(f"[PTReceiver] 正在监听 {self.listen_ip}:{self.port}...")
            conn, addr = server_socket.accept()
            print(f"[PTReceiver] 已连接 from {addr}")

            buffer = b''
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buffer += chunk

            pt_stream = io.BytesIO(buffer)
            obj = torch.load(pt_stream)
            print(f"[PTReceiver] 接收完成，共 {len(buffer)} 字节，成功反序列化对象")
            return obj
