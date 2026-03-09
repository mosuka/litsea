#!/bin/bash
#
# Split text into sentences using regex-based rules (bracket-aware).
#
# Reads text from stdin (one paragraph per line), splits each line into
# sentences, and writes one sentence per line to stdout.
# Empty lines in the input are skipped.
#
# This script does NOT split on punctuation inside brackets/quotes:
#   「」『』（）() \u201c\u201d \u2018\u2019
#
# Sentence boundaries (outside brackets only):
#   - CJK full-width punctuation: 。 ！ ？
#   - Western punctuation followed by whitespace or end of line: . ! ?
#
# Usage:
#   echo "彼は「こんにちは。太郎です。」と言った。" | bash scripts/split_sentences.sh -l ja
#   echo "First sentence. Second sentence." | bash scripts/split_sentences.sh -l en

usage() {
    echo "Usage: $0 [-h] [-l language]"
    echo "  -l language   Language code (e.g. ja, ko, zh, en). Currently unused;"
    echo "                the splitting rules are language-independent."
    echo "  -h            Show this help message."
    echo ""
    echo "Reads from stdin, writes one sentence per line to stdout."
    exit 1
}

while getopts "hl:" opt; do
    case "$opt" in
        h) usage ;;
        l) ;; # accepted but currently unused
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

# Split sentences using Perl with bracket-depth tracking.
# Punctuation inside brackets/quotes is preserved as-is.
perl -CS -Mutf8 -ne '
    chomp;
    next if /^\s*$/;

    my $line = $_;
    my $depth = 0;
    my $sentence = "";
    my @chars = split //, $line;

    for (my $i = 0; $i <= $#chars; $i++) {
        my $c = $chars[$i];
        $sentence .= $c;

        # Track bracket depth
        if ($c =~ /[「『（(\x{201C}\x{2018}]/) {
            $depth++;
            next;
        }
        if ($c =~ /[」』）)\x{201D}\x{2019}]/) {
            $depth-- if $depth > 0;
            next;
        }

        # Only split when outside brackets
        next if $depth > 0;

        # CJK sentence-ending punctuation: always split
        if ($c =~ /[。！？]/) {
            my $s = $sentence;
            $s =~ s/^\s+|\s+$//g;
            print "$s\n" if $s ne "";
            $sentence = "";
            next;
        }

        # Western sentence-ending punctuation: split if followed by space or end
        if ($c =~ /[.!?]/) {
            if ($i == $#chars || ($i < $#chars && $chars[$i+1] =~ /\s/)) {
                my $s = $sentence;
                $s =~ s/^\s+|\s+$//g;
                print "$s\n" if $s ne "";
                $sentence = "";
                next;
            }
        }
    }

    # Print remaining text
    $sentence =~ s/^\s+|\s+$//g;
    print "$sentence\n" if $sentence ne "";
'
