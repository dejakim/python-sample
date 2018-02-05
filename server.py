#!/usr/bin/env python
#-*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import argparse
import socket
import threading
import json

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from urlparse import urlparse

def handle_connection(client):
    req = unicode(client.recv(1024))
    print 'Received {}'.format(req)
    client.send(json.dump({'stat':'OK'}))
    client.close()

class Handler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        req = urlparse(self.path)
        print req.query
        self._set_headers()
        self.wfile.write("<html><body><h1>hi!</h1></body></html>")

    def do_HEAD(self):
        self._set_headers()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple python server')
    parser.add_argument('--port', '-p', default='8000', help='Port number')
    args = parser.parse_args()

    httpd = HTTPServer(('', int(args.port)), Handler)
    print 'Starting httpd...'
    httpd.serve_forever()

'''
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Create socket on port
    try:
        server.bind(('', int(args.port)))
    except socket.error as msg:
        print('Could not establish socket server')
        sys.exit()

    print('Socket server established')

    # Start listening on socket
    server.listen(10)

    # Receive data from client
    while True:
        client, address = server.accept()
        print 'Accepted connection from {}:{}'.format(address[0], address[1])
        client_handler = threading.Thread(
            target = handle_connection,
            args = (client_sock,) # without comma you'd get a... TypeError: handle_client_connection() argument after * must be a sequence, not _socketobject
        )
        client_handler.start()

    server.close()
'''
