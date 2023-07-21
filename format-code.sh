#!/bin/sh
for f in $(find src -name "*.py"); do
	black $f
	isort $f
done
