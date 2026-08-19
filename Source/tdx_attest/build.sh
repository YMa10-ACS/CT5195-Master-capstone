#!/usr/bin/env bash

go mod tidy
go build -o tdx-attest .

