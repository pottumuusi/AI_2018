#!/bin/bash

test_dir=$(cd $(dirname $0) && pwd)
cd $test_dir

readonly DEBUG="n"
readonly TEST_AMOUNT=2

# selected_python="python"
selected_python="python2"
search_agent="SearchAgent"

latest_run_file="latest_run.txt"
test_results_file="test_results.txt"
test_results_path="$test_dir/$test_results_file"
test_summary_file="test_summary.txt"
test_summary_path="$test_dir/$test_summary_file"
test_temp_file="test_temp.txt"
test_temp_path="$test_dir/$test_temp_file"

if [ "y" == "$DEBUG" ] ; then
	echo "test_dir is: $test_dir"
	echo "\$test_dir/\$test_results_file is: $test_dir/$test_results_file"
	echo "test_results_path is: $test_results_path"
fi

run_tests() {
	# Truncate previous test results
	echo "" > $test_results_file

	cd ../Code
	$selected_python pacman.py --frameTime 0 -l tinyMaze -p $search_agent
	ret=$?
	cat $latest_run_file >> "$test_results_path"
	echo "Ret is: $ret" >> "$test_results_path"

	$selected_python pacman.py --frameTime 0 -l mediumMaze -p $search_agent
	ret=$?
	cat $latest_run_file >> "$test_results_path"
	echo "Ret is: $ret" >> "$test_results_path"

	$selected_python pacman.py --frameTime 0 -l bigMaze -z .5 -p $search_agent
	ret=$?
	cat $latest_run_file >> "$test_results_path"
	echo "Ret is: $ret" >> "$test_results_path"
}

generate_summary_file() {
	echo "" > $test_summary_path
	cat $test_results_path | grep "Result is goal state" >> $test_summary_path
	# TODO create summary for directions from start to goal state
}

goal_state_results() {
	echo -e $(grep "Result is goal state" $test_summary_path)
}

evaluate_test() {
	test_no=$1
	success="y"
	fail_reason=""

	if [ "1" == "$test_no" ] ; then
		test_name="Goal state test"
		count=1

		grep "Result is goal state" $test_summary_path > $test_temp_path

		while read -r result ; do
			if [ -z "$(echo $result | grep "True")" ] ; then
				success="n"
				fail_reason="Search result was not goal state in case no. $count"
				break
			fi
			count=$(( $count + 1 ))
		done < "$test_temp_path"
	fi

	if [ "2" == "$test_no" ] ; then
		test_name="Correct direction list"
		success="n"
		fail_reason="Not implemented"
	fi

	if [ "y" == "$success" ] ; then
		echo "($test_name) OK"
	else
		echo "($test_name) FAIL, reason: $fail_reason"
	fi
}

print_test_results() {
	echo ""
	echo "==================== Test results ===================="
	for i in $(seq 1 $TEST_AMOUNT) ; do
		echo "Test $i -> $(evaluate_test $i)"
	done
	echo "==================== Test results ===================="
	echo ""
}

run_tests
generate_summary_file
print_test_results
