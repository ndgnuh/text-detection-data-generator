#!/bin/sh

for f in $(find . -name "*.py"); do
	black $f
	isort $f
done
