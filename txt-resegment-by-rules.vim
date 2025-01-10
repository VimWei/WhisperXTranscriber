normal ggVGJ
" %s/—/ /ge
" %s/-/ /ge
%s/  / /ge
%s/  / /ge
%s/\. /\.\r/ge
%s/\! /\!\r/ge
%s/? /\?\r/ge
%s/, and/,\rand/ge
%s/, because/,\rbecause/ge
%s/ because/\rbecause/ge
%s/^\(and\)\n\(because \)/\1 \2/ge
%s/, so/,\rso/ge
" %s/ so /\rso /ge
" %s/^\(and\)\n\(so \)/\1 \2/ge
%s/, then/,\rthen/ge
%s/ and then/\rand then/ge
%s/^\(and\)\n\(then \)/\1 \2/ge
%s/, but/,\rbut/ge
%s/ but /\rbut /ge
%s/, instead of/,\rinstead of/ge
%s/, which/,\rwhich/ge
%s/, what/,\rwhat/ge
%s/, when/,\rwhen/ge
%s/, where/,\rwhere/ge
%s/, why/,\rwhy/ge
%s/, how/,\rhow/ge
%s/; /;\r/ge
%s/^ //ge
silent! g/^$/d
w
