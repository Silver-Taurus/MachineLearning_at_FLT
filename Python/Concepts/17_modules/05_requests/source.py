''' Use of requests module '''

import requests
import os

def halt_reset():
    ''' Just a utility function '''
    while(True):
        response = input('\nPress Enter to continue: ') or '\n'
        if(response):
            break
    os.system('clear')


# Few attributes
R = requests.get('https://api.github.com/')
print(R.status_code)
print(R.headers['content-type'])
print(R.encoding)
print(R.text)
print(R.json())


halt_reset()


# Making a request

# Let's try getting a webpage, for ex: getting github's public timeline
R = requests.get('https://api.github.com/events')

# After requesting we had got a response
print(R.status_code)

# request's simple API means that all forms of HTTP request are obvious. For example:
# this is how you make an http post request
# R = requests.post('http://httpbin.org/post', data={'key': 'value'})
# Similarly
# R = requests.put('http://httpbin.org/put', data={'key': 'value'})
# R = requests.delete('http://httpbin.org/delete')
# R = requests.head('http://httpbin.org/head')
# R = requests.options('http://httpbin.org/options')


# Passing arguments in URLs
PAYLOAD = {'key1': 'value1', 'key2': 'value2'}
R = requests.get('https://httpbin.org/get', params=PAYLOAD)
print(R.url)
# Note that any dictionary key whose value is None will not be added to the URL's query string

# You can also add a list of items as a value.
PAYLOAD = {'key1': 'value1', 'key2': ['value2', 'value3']}
R = requests.get('https://httpbin.org/get', params=PAYLOAD)
print(R.url)


# Response Content
R = requests.get('https://api.github.com/events')
print(R.text)
print()
