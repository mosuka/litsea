# frozen_string_literal: true

require_relative 'litsea/version'

# Loads the compiled extension, preferring a version-specific build when
# rake-compiler produced one.
begin
  RUBY_VERSION =~ /(\d+\.\d+)/
  require "litsea/#{Regexp.last_match(1)}/litsea_ruby"
rescue LoadError
  require 'litsea/litsea_ruby'
end

# Word segmentation and POS tagging.
#
# See {Litsea::Segmenter} to get started; every class is defined by the
# native extension.
module Litsea
end
