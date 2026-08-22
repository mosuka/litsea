# frozen_string_literal: true

require 'mkmf'
require 'rb_sys/mkmf'

# The crate lives at the gem root (it is also a workspace member of the
# Litsea repository), so no ext_dir override is needed.
create_rust_makefile('litsea/litsea_ruby')
