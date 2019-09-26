#!/bin/bash

echo "First argument is $1 and the Second argument is $2"
HOST="$(hostname)"
MAC="$(ifconfig wifi0 | grep ether)"
echo "The MAC address for ${HOST} wifi is $(sed -e 's/ether/""/' ${MAC})"
