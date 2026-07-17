BEGIN {
    outfile = ARGV[2]
    ARGC--
    record = 0
    failed = 0
}

# CGAL 6 writes a blank line between the OFF counts and the first vertex.
# Count meaningful records instead of physical lines so comments and blank
# lines cannot shift the vertex/triangle boundary.
$1 == "#" || NF == 0 { next }

record == 0 {
    if ($1 != "OFF") {
        print "Not an OFF file" > "/dev/stderr"
        failed = 1
        exit 1
    }
    record++
    next
}

record == 1 {
    nv = $1
    nt = $2
    print nv > outfile
    print nt >> outfile
    record++
    next
}

record >= 2 && record < 2 + nv {
    print $1, $2, $3 >> outfile
    record++
    next
}

record >= 2 + nv && record < 2 + nv + nt {
    print $1, $2, $3, $4 >> outfile
    record++
    next
}

END {
    if (!failed && record != 2 + nv + nt) {
        print "OFF record count does not match its header" > "/dev/stderr"
        exit 1
    }
}
